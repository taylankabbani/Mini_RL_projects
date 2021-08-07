import gym
from gym import envs 
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from utils import available_env

class QLearning:
    
    def __init__(self, env: str):
        self._env = gym.make(env)
        # Q-learning parameteres
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
            exploration_rate =  max(min_rate, a * current_episode + b)
        else:
            return 'Decay strategy not recognized'

        return exploration_rate



    def train(self, n_episods: int, lr_rate: float, gamma: float, policy: str):
        '''The training loop where the agent interact with the env
        '''
        print("\n\n****** Start Training ******\n\n")
        avg_reward = 0
        counter = 1
        rewards_per_1000_eps = []

        exploration_rate = 1 # Initially is high to try all possible actions

        print("***** Average rewards per 1000 episodes *****")
        for episode in range(n_episods):
            #initial state
            state = self._env.reset()
            done = False
            episodic_reward = 0

            # Till the end of the episode
            while not done:
                # Diversification/exploitation
                random_x = float(np.random.uniform(0,1))
                if exploration_rate > random_x :
                    #Explore
                    action = self._env.action_space.sample()

                else:
                    # Exploit
                    action = np.argmax(self.Q_tabel[state, :])
                
                # Take action, observe next state and reward received 
                next_state, reward, done, info = self._env.step(action)

                # Update Q_tabel
                q_value = float(self.Q_tabel[state, action])
                TD_error = float(reward + gamma * np.max(self.Q_tabel[next_state, :]) - q_value)
                self.Q_tabel[state, action] = float(q_value + (lr_rate * TD_error))

                state = next_state
                episodic_reward += reward
            
            # Decrease Exploration rate after each episode
            exploration_rate = self._annealing_policy(current_episode=episode, decay=policy, nb_episodes=n_episods)

            avg_reward += episodic_reward
            
            
            # avg reward per 1000 episode
            if episode//1000 == counter:
                rewards_per_1000_eps.append(avg_reward)
                print(episode, ' : ', avg_reward/1000)
                avg_reward = 0
                counter += 1

        print("\n\n****** Training Finished ******\n\n")
        plt.plot(rewards_per_1000_eps)
        plt.show()








if __name__ == '__main__':
    test = QLearning(env="Taxi-v3")
    test.train(n_episods=40000, lr_rate=0.1, gamma=0.99, policy='linear')