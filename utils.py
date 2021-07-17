import gym
import regex as re


def available_env():
    """Show available environment"""
    envs_list = []
    for game in gym.envs.registry.all():
        pattern = r'(?i)[A-Z0-9-]+-v0(?=\))'
        env_name = re.search(pattern, str(game))
        if env_name is None:
            pass
        else:
            envs_list.append(env_name.group(0))
    return envs_list
