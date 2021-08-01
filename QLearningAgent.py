import gym 
import numpy as np
from IPython.display import clear_output
from utils import available_env

# class QLearning:
#     def __init__(self):
#         env = 's'
    
#     def select_env(self):
#         envs_list = available_env(Agent=QLearning)

# for env in gym.envs.registry.all():
#     print(env)

env = gym.make("CartPole")
actions = env.action_space.n
states = env.observation_space.n
print("Number of actions: ",actions, "\nNumber of states: ", states)