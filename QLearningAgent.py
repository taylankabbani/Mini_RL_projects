import gym
from gym import envs 
import numpy as np
from IPython.display import clear_output
from utils import available_env

class QLearning:
    
    def __init__(self, env: str, max_steps_per_episode: int, gamma: float, lr_rate: float):
        self._env = gym.make(env)
        # Q-learning parameteres
        self.max_steps_per_episode = max_steps_per_episode
        self.gamma = gamma
        self.lr_rate = lr_rate
        self.Q_tabel = self.creat_Q_tabel()

    def creat_Q_tabel(self):
        actions = self._env.action_space.n
        states = self._env.observation_space.n
        # Creating a Q-table where rows are states and columns are actions
        # Note that for other environments where states are infint (e.g. state is an array of pixels) the 
        # number of states is not avialable rather you can get the shape of the state
        Q_tabel = np.zeros((states, actions))
        return Q_tabel
    
    def _annealing_policy(self, current_episode, decay: str, nb_episodes: int):
        '''Annealing policy with exponential/linear decaying
        returns the exploration rate based on the episode number
        '''
        max_rate = 1
        min_rate = 0.01
        decay_rate = 0.001
        if decay== 'exp':
            exploration_rate = min_rate + (max_rate - min_rate)* np.exp(-decay_rate*current_episode)
        
        elif decay == 'linear':
            a = -(max_rate - min_rate)/nb_episodes
            b = max_rate
            exploration_rate = lambda episode: max(min_rate, a * current_episode + b)
        else:
            return 'Decay strategy not recognized'

        return exploration_rate



    def train(self, n_episods: int):
        '''The training loop where the agent interact with the env
        '''
        for episode in range(n_episods):
            #initial state
            state_0 = self._env.reset()
            done = False
            current_reward = 0
            exploration_rate = 1 # Initially is high to try all possible actions
            # Till the end of the episode
            while not done:
                # Diversification/exploitation
                random_x = float(np.random.uniform(0,1))
                if exploration_rate > random_x :
                    pass
                else:
                    pass
                





### Debug ###
test = QLearning(env="Taxi-v3", n_episods="me", max_steps_per_episode='a', gamma=0.1, lr_rate=0.5)
print(test.Q_tabel)
# print(test._env)
# print(test.actions)
# test.select_env()
    # def select_env(self):
    #     envs_list = available_env(Agent='toy')
    #     [print(env) for env in envs_list]
    #     user_input = input('Choose a game to play: ')
    #     while user_input not in envs_list:
    #         user_input = input("\nThe game does not exist.\n\n Choose a game to play: ")
    #     env = gym.make(user_input)
    #     return env
# for env in gym.envs.registry.all():
#     print(env)

# env = gym.make("Reacher-v2")
# actions = env.action_space.n
# states = env.observation_space.n
# print("Number of actions: ",actions, "\nNumber of states: ", states)
# envs_list = available_env(Agent='toy')
# for i in envs_list:
#     print(i)