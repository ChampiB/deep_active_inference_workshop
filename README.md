# Deep Active Inference Workshop

This repository contains the code presented during the symposium on amortising active inference (Wednesday September 14th).

## 1. Installation

To install the dependencies of the project, execute the following command:
```
pip install -r requirements.txt
```

Install pillow packages (for Python 3 on Ubuntu):
```
sudo apt-get install python3-pil python3-pil.imagetk
```

## 2. Training a model

Train a model requires three steps. First, edit the configuration to train the desired model. Second, tell the project where to find the dSprites dataset. Third, run the training script.

### 2.1. The configuration

The configuration of the project can be found in the directory named `config`, and is based on Hydra: https://hydra.cc/docs/intro/.

The file `config/training.yaml` allow to choose which agent to train, as well as the number of training iterations to perform, etc...

The files `config/agent/CHMM.yaml` and `config/agent/DQN.yaml` allows to modify the model specific hyper-parameters such as the discount factor, the size of the latent space, etc...

### 2.2. The environment variable

To tell the project where to find the dSprites dataset, the environment variable `DATA_DIRECTORY` must be set as follows:
```
export DATA_DIRECTORY=/path/to/data/directory
```

The dSprites dataset is provided in the directory `data`.

### 2.3. Running the script

Finally, to train the model, run the following command:
```
python3 ./env_training.py
```

## 3. Project extension

While this repository contains a simple implementation of DQN and deep active inference agent, a more complete version can be found here:
https://github.com/ChampiB/Challenges_Deep_Active_Inference