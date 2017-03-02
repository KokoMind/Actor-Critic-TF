import numpy as np
import tensorflow as tf
import itertools
import os
from model import Actor, Critic


class Agent:
    def __init__(self, sess, config, environment):
        # Get the session, config, environment, and create a replaymemory
        self.sess = sess
        self.config = config
        self.environment = environment

        self.init_dirs()
        self.init_cur_epsiode()
        self.init_global_step()
        self.init_summaries()

        # Intialize the graph which contain 2 Networks Actor and Critic
        self.actor = Actor(sess, self.environment.n_actions, self.environment.state_shape, config)
        self.critic = Critic(sess, self.environment.state_shape, config)

        # To initialize all variables
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        self.saver = tf.train.Saver(max_to_keep=10)
        self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

        if config.is_train and not config.cont_training:
            pass
        elif config.is_train and config.cont_training:
            self.load()
        elif config.is_play:
            self.load()
        else:
            raise Exception("Please Set proper mode for training or playing")

    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)

    def save(self):
        self.saver.save(self.sess, self.checkpoint_dir, self.global_step_tensor)

    def init_dirs(self):
        # Create directories for checkpoints and summaries
        self.checkpoint_dir = os.path.join(self.config.experiment_dir, "checkpoints/")
        self.summary_dir = os.path.join(self.config.experiment_dir, "summaries/")

    def init_cur_epsiode(self):
        """Create cur episode tensor to totally save the process of the training"""
        with tf.variable_scope('cur_episode'):
            self.cur_episode_tensor = tf.Variable(-1, trainable=False, name='cur_epsiode')
            self.cur_epsiode_input = tf.placeholder('int32', None, name='cur_episode_input')
            self.cur_episode_assign_op = self.cur_episode_tensor.assign(self.cur_epsiode_input)

    def init_global_step(self):
        """Create a global step variable to be a reference to the number of iterations"""
        with tf.variable_scope('step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

    def init_summaries(self):
        """Create the summary part of the graph"""
        with tf.variable_scope('summary'):
            self.summary_placeholders = {}
            self.summary_ops = {}
            self.scalar_summary_tags = ['episode.total_reward', 'episode.length', 'evaluation.total_reward', 'evaluation.length', 'epsilon']
            for tag in self.scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

    def add_summary(self, summaries_dict, step):
        """Add the summaries to tensorboard"""
        summary_list = self.sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()],
                                     {self.summary_placeholders[tag]: value for tag, value in summaries_dict.items()})
        for summary in summary_list:
            self.summary_writer.add_summary(summary, step)
        self.summary_writer.flush()

    def take_action(self, state):
        """Take the action"""
        action_probs = self.actor.predict(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def observe(self, action):
        """Function that observe the new state, reward"""
        return self.environment.step(action)

    def train_episodic(self):
        """Train the agent in episodic techniques"""

        for cur_episode in range(self.cur_episode_tensor.eval(self.sess) + 1, self.config.num_episodes, 1):

            # Save the current checkpoint
            self.save()

            # Update the Cur Episode tensor
            self.cur_episode_assign_op.eval(session=self.sess, feed_dict={self.cur_epsiode_input: self.cur_episode_tensor.eval(self.sess) + 1})

            state = self.environment.reset()
            total_reward = 0

            # Take steps in the environment untill terminal state of epsiode
            for t in itertools.count():

                # Update the Global step
                self.global_step_assign_op.eval(session=self.sess, feed_dict={self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})

                # Take an action
                action = self.take_action(state)
                next_state, reward, done = self.observe(self.environment.valid_actions[action])

                # Calculate the TD Target
                value_next = self.critic.predict(next_state)
                td_target = reward + self.config.discount_factor * value_next
                td_error = td_target - self.critic.predict(state)

                # Update the Critic
                self.critic.update(state, td_target)

                # Update the Actor
                # using the td error as our advantage estimate
                # TODO Research about the best advantage estimate
                self.actor.update(state, action, td_error)

                total_reward += reward

                if done:  # IF terminal state so exit the episode
                    # Add summaries to tensorboard
                    summaries_dict = {'episode.total_reward': total_reward,
                                      'episode.length': t}
                    self.add_summary(summaries_dict, self.global_step_tensor.eval(self.sess))
                    break

                state = next_state

        print("Training Finished")
