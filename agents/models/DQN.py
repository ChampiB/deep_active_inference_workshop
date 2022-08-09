from agents.networks.NeuralNetworks import PolicyNetwork
from agents.EpsilonGreedy import EpsilonGreedy
from torch import nn, unsqueeze
from torch.utils.tensorboard import SummaryWriter
from agents.Checkpoint import Checkpoint
from datetime import datetime
from agents.ReplayBuffer import ReplayBuffer, Experience
import torch
from copy import deepcopy
from agents import Optimizers
from singletons.Device import Device


#
# Class implementing a Deep Q-network.
#
class DQN:

    #
    # Instances construction.
    #

    def __init__(
            self, image_shape, n_steps_between_synchro, discount_factor, lr, tensorboard_dir, n_actions=4,
            steps_done=0, **_
    ):
        """
        Constructor
        :param image_shape: the shape of input images
        :param n_actions: the number of actions
        :param n_steps_between_synchro: the number of steps between two synchronisations of the target and the critic
        :param discount_factor: the value by which future rewards are discounted
        :param lr: the learning rate of the Q-network
        :param tensorboard_dir: the directory in which tensorboard's files will be written
        :param steps_done: the number of training step done so far
        """

        # The hyper-parameters.
        self.n_steps_between_synchro = n_steps_between_synchro
        self.discount_factor = discount_factor
        self.lr = lr
        self.tensorboard_dir = tensorboard_dir
        self.action_selection = EpsilonGreedy()

        # Create the summary writer for monitoring with TensorBoard.
        self.writer = SummaryWriter(tensorboard_dir)

        # Number of training steps performed to date and total rewards gather so far.
        self.steps_done = steps_done
        self.total_rewards = 0

        # Neural networks.
        self.policy = PolicyNetwork(image_shape, n_actions)
        self.target = deepcopy(self.policy)
        self.target.eval()

        # Ensure models are on the right device.
        Device.send([self.policy, self.target])

        # Create the agent's optimizers.
        self.optimizer = Optimizers.get_adam([self.policy], lr)

        # Create the replay buffer.
        self.buffer = ReplayBuffer()

    #
    # Main functionalities.
    #

    def train(self, env, config):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        :param config: the hydra configuration
        :return: nothing
        """

        # Retrieve the initial observation from the environment.
        obs = env.reset()

        # Render the environment.
        if config["display_gui"]:
            env.render()

        # Train the agent.
        print("Start the training at {time}".format(time=datetime.now()))
        while self.steps_done < config["n_training_steps"]:

            # Select an action.
            action = self.step(obs)

            # Execute the action in the environment.
            old_obs = obs
            obs, reward, done, _ = env.step(action)

            # Add the experience to the replay buffer.
            self.buffer.append(Experience(old_obs, action, reward, done, obs))

            # Perform one iteration of training if needed.
            if len(self.buffer) >= config["buffer_start_size"]:
                self.learn(config)

            # Save the agent if needed.
            if self.steps_done % config["checkpoint"]["frequency"] == 0:
                self.save(config)

            # Render the environment if needed.
            if config["enable_tensorboard"]:
                self.total_rewards += reward
                self.writer.add_scalar("Rewards", self.total_rewards, self.steps_done)
            if config["display_gui"]:
                env.render()

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()

            # Increase the number of iterations performed.
            self.steps_done += 1

        # Close the environment.
        env.close()

    def step(self, obs):
        """
        Choose the next action to perform in the environment
        :param obs: the current observation
        :return: the action to perform
        """

        # Create a 4D tensor from a 3D tensor by adding a dimension of size one.
        obs = torch.unsqueeze(obs, dim=0)

        # Select an action to perform in the environment.
        return self.action_selection.select(self.policy(obs), self.steps_done)

    def learn(self, config):
        """
        Learn the parameters of the agent
        :param config: the hydra configuration
        :return: nothing
        """

        # Synchronize the target network with the policy network if needed
        if self.steps_done % self.n_steps_between_synchro == 0:
            self.target = deepcopy(self.policy)
            self.target.eval()

        # Sample the replay buffer.
        obs, actions, rewards, done, next_obs = self.buffer.sample(config["batch_size"])

        # Compute the policy network's loss function.
        loss = self.compute_loss(config, obs, actions, rewards, done, next_obs)

        # Perform one step of gradient descent on the other networks.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_loss(self, config, obs, actions, rewards, done, next_obs):
        """
        Compute the loss function used to train the policy network
        :param config: the hydra configuration
        :param obs: the observations made at time t
        :param actions: the actions performed at time t
        :param rewards: the rewards received at time t + 1
        :param done: whether the episode ended after performing action a_t
        :param next_obs: the observations made at time t + 1
        :return: the policy loss
        """

        # Compute the q-values of the current state and action
        # as predicted by the policy network, i.e. Q(s_t, a_t).
        policy_prediction = self.policy(obs).gather(dim=1, index=unsqueeze(actions.to(torch.int64), dim=1))

        # For each batch entry where the simulation did not stop, compute
        # the value of the next states, i.e. V(s_{t+1}). Those values are computed
        # using the target network.
        future_values = torch.zeros(config["batch_size"]).to(Device.get())
        future_values[torch.logical_not(done)] = self.target(next_obs[torch.logical_not(done)]).max(1)[0]
        future_values = future_values.detach()

        # Compute the expected Q values
        total_values = rewards + future_values * self.discount_factor

        # Compute the loss function.
        loss = nn.SmoothL1Loss()
        loss = loss(policy_prediction, total_values.unsqueeze(1))

        # Print debug information, if needed.
        if config["enable_tensorboard"]:
            self.writer.add_scalar("Q-values loss", loss, self.steps_done)

        return loss

    def predict(self, obs):
        """
        Do one forward pass using given observation.
        :return: the outputs of the policy network
        """
        return self.policy(obs)

    #
    # Saving and reloading the model.
    #

    @staticmethod
    def load_constructor_parameters(tb_dir, checkpoint, training_mode=True):
        """
        Load the constructor parameters from a checkpoint
        :param tb_dir: the path of tensorboard directory
        :param checkpoint: the checkpoint from which to load the parameters
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :return: a dictionary containing the constructor's parameters
        """
        return {
            "lr": checkpoint["lr"],
            "action_selection": Checkpoint.load_object_from_dictionary(checkpoint, "action_selection"),
            "discount_factor": checkpoint["discount_factor"],
            "image_shape": checkpoint["image_shape"],
            "tensorboard_dir": tb_dir,
            "n_steps_between_synchro": checkpoint["n_steps_between_synchro"],
            "steps_done": checkpoint["steps_done"],
            "n_actions": checkpoint["n_actions"]
        }

    def save(self, config):
        """
        Create a checkpoint file allowing the agent to be reloaded later
        :param config: the hydra configuration.
        :return: nothing
        """

        # Create directories and files if they do not exist.
        checkpoint_file = config["checkpoint"]["file"]
        Checkpoint.create_dir_and_file(checkpoint_file)

        # Save the model.
        torch.save({
            "agent_module": str(self.__module__),
            "agent_class": str(self.__class__.__name__),
            "image_shape": config["images"]["shape"],
            "n_actions": config["agent"]["n_actions"],
            "policy_net_state_dict": self.policy.state_dict(),
            "policy_net_module": str(self.policy.__module__),
            "policy_net_class": str(self.policy.__class__.__name__),
            "steps_done": self.steps_done,
            "lr": self.lr,
            "tensorboard_dir": self.tensorboard_dir,
            "discount_factor": self.discount_factor,
            "n_steps_between_synchro": self.n_steps_between_synchro,
            "action_selection": dict(self.action_selection),
        }, checkpoint_file)
