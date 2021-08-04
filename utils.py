import gym
import regex as re


def available_env(Agent='Atari'):
    """Show available environment"""
    envs_list = []
    pattern = r'(?i)[A-Z0-9-]+.(?=\))'
    if Agent == 'toy':
        envs_list = ["FrozenLake-v0", "FrozenLake8x8-v0", "CliffWalking-v0", "NChain-v0", "Roulette-v0", "Taxi-v3"]
        return envs_list
    for game in gym.envs.registry.all():
        env_name = re.search(pattern, str(game))
        if env_name is None:
            pass
        else:
            envs_list.append(env_name.group(0))
    return envs_list


