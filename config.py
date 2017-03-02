"""This file is to put any configuration to our classes"""


class EnvConfig(object):
    env_name = 'Breakout-v0'
    record_video_every = 5


class AgentConfig(object):
    discount_factor = 0.99
    evaluate_every = 25
    evaluation_episodes = 5


class ModelConfig(object):
    name = "DQN_Dragon"
    learning_rate = 0.00025


class Experiment1(EnvConfig, AgentConfig, ModelConfig):
    is_train = True
    cont_training = True
    is_play = False
    num_episodes = 10000

    experiment_dir = "./experiment_1/"


def get_config():
    return Experiment1
