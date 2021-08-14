""" Building an agent that will use DQN algorithm to play atari games """

import gym
import random
from gym import envs
from keras.backend_config import epsilon
import numpy as np
import flappy_bird_gym
from collections import deque
from keras.optimizers import rmsprop_v2 as RMS
from tensorflow.python.keras import models
from utils import build_DenseNet



class DQNAgent:
    def __init__(self, env: str, memory_size: int, discount_factor: float, learning_rate:float):
        # Environment Variables
        self._env = self._set_env(env)
        self.state_space = self._env.observation_space.shape[0]
        self.action_space = self._env.action_space.n
        # The Table (Database) of action/state values
        self.memory = deque(maxlen=memory_size)

        # Learning hyperparameters
        self.gamma = discount_factor
        self.alpha = learning_rate 


    def _set_env(self, env:str):
        if env == "FlappyBird":
            return flappy_bird_gym.make("FlappyBird-v0")
        else:
            return gym.make(env)
        

    def _deep_model(self):
        """Building the NN network"""
        state_shape = self._env.observation_space.shape[0]
        num_action = self._env.action_space.n
        model = build_DenseNet(nn_input=state_shape, nn_output=(num_action))
        """Compile the network"""
        model.compile(loos = 'mse',
            optimizer=RMS(lr=0.0001, rho=0.95, epsilon=0.01), metrics=['accuracy'])

        # print(height, width, channels)
        # nn_model = Sequential([
        #     Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu",
        #            input_shape=(self.batch_size, height, width, channels)),
        #     Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu"),
        #     Flatten(),
        #     Dense(units=512, activation="relu"),
        #     Dense(units=256, activation="relu"),
        #     Dense(units=self.actions, activation="linear")
        # ])
        # return nn_model

    # def _build_agent(self):
    #     """The policies the RL agent will follow to learn Q-value,  as it's off-policy, the agent will use one greedy
    #     policy to always choose the greedy action (Q-value) and another policy that will break the greedy action
    #     selection by rate of epsilon
    #     """
    #     pass

if __name__ == "__main__":
    test = DQNAgent(env='FlappyBird', memory_size=2000, discount_factor=0.99, learning_rate=0.5)
    test._build_nn_model()