""" Let's Begin the action :P  """

import tensorflow as tf
# from environment import Environment
from agent import Agent
from config import get_config
from utils import create_dirs


def main():
    # Reset the graph
    tf.reset_default_graph()

    # Get the Config of the program and init the dirs
    config = get_config()
    create_dirs(config.experiment_dir)

    # Create the Session of the graph
    sess = tf.Session()

    # env = Environment(sess, config)
    # evaluation_env = Environment(sess, config, evaluation=True)

    # wasted = Agent(sess, config, env, evaluation_env)
    #
    # if config.is_train:
    #     try:
    #         wasted.train_episodic()
    #     except KeyboardInterrupt:
    #         wasted.save()
    # elif config.is_play:
    #     wasted.play()
    # else:
    #     raise Exception("Please select a proper mode for our wasted agent")

    sess.close()


if __name__ == '__main__':
    main()
