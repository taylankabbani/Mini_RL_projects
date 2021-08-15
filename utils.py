import gym
from numpy.matrixlib import defmatrix
import regex as re
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D


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
    envs_list.append("FlappyBird")
    return envs_list

def build_ConvNet(nn_input, nn_output):
    """Build Convolution neural network to process game input pixels.
    This is suitable for RGB version of the Atari games"""

    ConvNet_model = Sequential([
        Conv2D(filter=32,  kernel_size=(8,8), strides=(2,2), activation="relu", input_shape=nn_input),
        MaxPool2D(pool_size=(2,2)),
        Conv2D(filter=64,  kernel_size=(8,8), strides=(2,2), activation="relu"),
        MaxPool2D(pool_size=(2,2)),
        Conv2D(filter=128,  kernel_size=(8,8), strides=(2,2), activation="relu"),
        MaxPool2D(pool_size=(4,4)),
        Flatten(),
        Dense(units=521, activation="relu"),
        Dense(units=256, activation="relu"),
        Dense(units=nn_output, activation="linear")
    ])
    return ConvNet_model


def build_DenseNet(nn_input, nn_output):
    """Build Dense neural network to process simple numerical information about the game's state as observations.This is suitable for Ram version of the Atari games"""
    nn_model = Sequential([
        Dense(units=512, input_shape= nn_input, activation="relu", kernel_initializer='he_uniform'),
        Dense(units=256, activation='relu', kernel_initializer='he_uniform'),
        Dense(units=64, activation='relu', kernel_initializer='he_uniform'),
        Dense(units=nn_output, activation='relu', kernel_initializer='he_uniform')
    ])
    return nn_model

    


