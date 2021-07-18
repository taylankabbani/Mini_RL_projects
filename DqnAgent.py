""" Building an agent that will use DQN algorithm to play atari games  """

import gym
from utils import available_env
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent  # The DQN algorithm (agent)
from rl.memory import SequentialMemory  # The Tabular-like structure the agent will use to learn the Q-values
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


class Dqn:
    """Building the NN network"""

    def __init__(self, game_name: str, batch_size=2):
        self._env = gym.make(game_name)
        self.batch_size = batch_size
        self.actions = self._env.action_space.n

    def _build_nn_model(self):
        height, width, channels = self._env.observation_space.shape
        nn_model = Sequential([
            Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu",
                   input_shape=(self.batch_size, height, width, channels)),
            Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu"),
            Flatten(),
            Dense(units=512, activation="relu"),
            Dense(units=256, activation="relu"),
            Dense(units=self.actions, activation="linear")
        ])
        return nn_model

    def _build_agent(self):
        """The policies the RL agent will follow to learn Q-value,  as it's off-policy, the agent will use one greedy
        policy to always choose the greedy action (Q-value) and another policy that will break the greedy action
        selection by rate of epsilon
        """
        pass

