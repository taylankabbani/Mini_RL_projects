import gym
import regex as re


def available_env(Agent):
    """Show available environment"""
    envs_list = []
    if Agent == 'Dumb':
        pattern = r'(?i)[A-Z0-9-]+.(?=\))'
    elif Agent == 'QLearning':
        pass
    elif Agent == 'Dqn':
        pass
    for game in gym.envs.registry.all():
        env_name = re.search(pattern, str(game))
        if env_name is None:
            pass
        else:
            envs_list.append(env_name.group(0))
    return envs_list


