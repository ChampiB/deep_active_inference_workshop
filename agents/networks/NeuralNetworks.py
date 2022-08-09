import torch
from torch.nn.functional import one_hot
from math import prod
from agents.networks.layers.DiagonalGaussian import DiagonalGaussian as Gaussian
from torch import nn, zeros, cat


#
# Class implementing a critic network modeling the cost of each action given a state.
#
class CriticNetwork(nn.Module):

    def __init__(self, n_states, n_actions):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param n_actions: the number of allowable actions.
        """

        super().__init__()

        # Create the critic network.
        self.__net = nn.Sequential(
            nn.Linear(n_states, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions),
        )

    def forward(self, states):
        """
        Forward pass through the critic network.
        :param states: the input states.
        :return: the cost of performing each action in that state.
        """
        return self.__net(states)


#
# Implement the policy network that compute the q-values.
#
class PolicyNetwork(nn.Module):

    def __init__(self, image_shape, n_actions):
        """
        Constructor.
        :param image_shape: the shape of the input images.
        :param n_actions: the number of actions.
        """

        super().__init__()

        # Create convolutional part of the policy network.
        self.__conv_net = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
        )
        self.__conv_output_shape = self.__conv_output_shape(image_shape)
        self.__conv_output_shape = self.__conv_output_shape[1:]
        conv_output_size = prod(self.__conv_output_shape)

        # Create the linear part of the policy network.
        self.__linear_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=1),
        )

        # Create the full policy network.
        self.__policy_net = nn.Sequential(
            self.__conv_net,
            self.__linear_net
        )

    def __conv_output_shape(self, image_shape):
        """
        Compute the shape of the features output by the convolutional encoder.
        :param image_shape: the shape of the input image.
        :return: the shape of the features output by the convolutional encoder.
        """
        image_shape = list(image_shape)
        image_shape.insert(0, 1)
        input_image = zeros(image_shape)
        return self.__conv_net(input_image).shape

    def forward(self, x):
        """
        Compute the q-values of each possible actions.
        :param x: an observations from the environment.
        :return: the q-values.
        """
        return self.__policy_net(x)


#
# Class implementing a transition network modeling the temporal transition between hidden state.
#
class TransitionNetwork(nn.Module):

    def __init__(self, n_states, n_actions):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param n_actions: the number of allowable actions.
        """

        super().__init__()

        # Create the transition network.
        self.__net = nn.Sequential(
            nn.Linear(n_states + n_actions, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            Gaussian(100, n_states)
        )

        # Remember the number of actions.
        self.n_actions = n_actions

    def forward(self, states, actions):
        """
        Forward pass through the transition network.
        :param states: the input states.
        :param actions: the input actions.
        :return: the mean and log of the variance of the Gaussian over hidden state.
        """
        actions = one_hot(actions.to(torch.int64), self.n_actions)
        x = cat((states, actions), dim=1)
        return self.__net(x)


#
# Class implementing a convolutional encoder network for 64 by 64 images.
#
class EncoderNetwork(nn.Module):

    def __init__(self, n_states, image_shape):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param image_shape: the shape of the input images.
        """

        super().__init__()

        # Create the convolutional encoder network.
        self.__conv_net = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, (4, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (4, 4), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (2, 2), stride=(2, 2), padding=1),
            nn.ReLU(),
        )
        self.__conv_output_shape = self.__conv_output_shape(image_shape)
        self.__conv_output_shape = self.__conv_output_shape[1:]
        conv_output_size = prod(self.__conv_output_shape)

        # Create the linear encoder network.
        self.__linear_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            Gaussian(256, n_states)
        )

        # Create the full encoder network.
        self.__net = nn.Sequential(
            self.__conv_net,
            self.__linear_net
        )

    def __conv_output_shape(self, image_shape):
        """
        Compute the shape of the features output by the convolutional encoder.
        :param image_shape: the shape of the input image.
        :return: the shape of the features output by the convolutional encoder.
        """
        image_shape = list(image_shape)
        image_shape.insert(0, 1)
        input_image = zeros(image_shape)
        return self.__conv_net(input_image).shape

    def forward(self, x):
        """
        Forward pass through this encoder.
        :param x: the input.
        :return: the mean and logarithm of the variance of the Gaussian over latent variables.
        """
        return self.__net(x)


#
# Class implementing a deconvolution decoder network for 64 by 64 images.
#
class DecoderNetwork(nn.Module):

    def __init__(self, n_states, image_shape):
        """
        Constructor.
        :param n_states: the number of hidden variables, i.e., number of dimension in the Gaussian.
        :param image_shape: the shape of the input images.
        """

        super().__init__()

        # Create the de-convolutional network.
        self.__lin_net = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Linear(256, 1600),
            nn.ReLU(),
        )
        self.__up_conv_net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (4, 4), stride=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(0, 0), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (4, 4), stride=(2, 2), padding=(0, 0), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_shape[0], (4, 4), stride=(1, 1), padding=(0, 0), output_padding=(0, 0)),
        )

    def forward(self, x):
        """
        Compute the shape parameters of a product of beta distribution.
        :param x: a hidden state.
        :return: the shape parameters of a product of beta distribution.
        """
        x = self.__lin_net(x)
        x = torch.reshape(x, (x.shape[0], 64, 5, 5))
        return self.__up_conv_net(x)
