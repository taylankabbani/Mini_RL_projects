""" Building an agent that will use DQN algorithm to play atari games """

import gym
import random
from gym import envs
from keras.backend_config import epsilon
from keras.optimizer_v2.rmsprop import RMSProp
import numpy as np
import flappy_bird_gym
from collections import deque
import utils



class DQNAgent:
    def __init__(self, env: str, memory_size: int, discount_factor: float, learning_rate:float):
        # Environment Variables
        self._env = self._set_env(env)
        self.state_space = self._env.observation_space.shape[0]
        self.action_space = self._env.action_space.n
        self.memory = deque(maxlen=memory_size)

        # Learning hyperparameters
        self.gamma = discount_factor
        self.alpha = learning_rate 

        self.model = self._build_deep_model()

        self.last_state = None
        self.last_action = None

    def _set_env(self, env:str):
        if env == "FlappyBird":
            return flappy_bird_gym.make("FlappyBird-v0")
        else:
            return gym.make(env)
        

    def _build_deep_model(self):
        """Building the NN network"""
        state_shape = self._env.observation_space.shape
        num_action = self._env.action_space.n
        model = utils.build_DenseNet(nn_input= state_shape, nn_output=(num_action))
        """Compile the network"""
        model.compile(optimizer= RMSProp(learning_rate=0.0001, rho=0.95, epsilon=0.01), loss = 'mse', 
            metrics=['accuracy'])
        return model

    def actor(self, state):
        """The actor which is responsible of taking actions and exploring the env"""
        # reshaping state to be fed to the neural network
        state = np.reshape(state, (1, self.state_space))
        _action_values = (self.model(state))
        pass
    
    def agent_start(self):
         """The first method called when the experiment starts, called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            self.last_action [int] : The first action the agent takes.
        """




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
    test.actor((2,55))