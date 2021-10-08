""" Building an agent that will randomly play atari games  """
import gym
from utils import available_env


# The agent-environment interaction loop
def interact(nb_episodes, game_name):
    # Create SpaceInvaders env
    env = gym.make(game_name)
    avg_score = 0
    for episode in range(nb_episodes):
        state_0 = env.reset()
        done = False  # When true => the agent has lost (the end of an episode)
        score = 0
        while not done:
            env.render()  # To show how the agent interact with the env
            action = env.action_space.sample()  # take a random action
            # returns the observations ensued from the agent-env interaction
            state, reward, done, info = env.step(action)
            score += reward
        avg_score += score
        print(f"Episode: {episode}, Score: {score}")
    print(f"Avg Score in {nb_episodes} episodes: {avg_score / nb_episodes} ")
    env.close()


if __name__ == '__main__':
    envs_list = available_env()
    use_input_1 = 0
    while use_input_1 not in envs_list:
        print('Refer to https://gym.openai.com/envs/#atari')
        use_input_1 = input("Choose a game to play: (press 0 to check available games)\t")

        if use_input_1 == '0':
            [print(i) for i in envs_list]
    rounds = int(input("Number of episodes to play:\t"))
    print('#' * 20, f'Playing {use_input_1} for {rounds} rounds', '#' * 20)
    interact(nb_episodes=rounds, game_name=use_input_1)