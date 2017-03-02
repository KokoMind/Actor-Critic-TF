import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
import os


class Environment(object):
    """Wrapping the gym environment"""

    def __init__(self, sess, config, evaluation=False):
        """
        state_processor_params = { "resize_shape": (h, w),
                    "crop_box": (y1, x1, y2, x2),
                    "rgb": False,
                    "frames_num": 1 }
        """
        self.sess = sess
        self._env = gym.envs.make(config.env_name)
        self._monitor_path = os.path.join(config.experiment_dir, "monitor/")
        self._valid_actions = [x for x in range(self.n_actions)]

        if evaluation:
            self._env = wrappers.Monitor(self._env, self._monitor_path, resume=True,
                                         video_callable=lambda count: count % config.record_video_every == 0)

        self._init_state_processor(config.state_processor_params)

    def reset(self):
        state = self._env.reset()
        state = self._state_processor(state)
        if self._frames_num > 1:
            state = np.squeeze(state)
            self._states_stack = np.stack([state] * self._frames_num, axis=2)
            return self._states_stack
        else:
            return state

    def step(self, action):
        next_state, reward, done, _ = self._env.step(action)
        next_state = self._state_processor(next_state)
        if self._frames_num > 1:
            next_state = np.concatenate((next_state, self._states_stack[:, :, :self._frames_num - 1]), axis=2)
            self._states_stack = next_state
        return next_state, reward, done

    def sample_action(self):
        return np.random.choice(self._env.action_space.n)

    def submit(self, api_key):
        gym.upload(self._monitor_path, api_key=api_key)

    def _init_state_processor(self, state_processor_params):
        with tf.name_scope("state_processor"):
            h, w, c = self._env.observation_space.shape
            self._input_state = tf.placeholder(shape=[h, w, c], dtype=tf.uint8, name='input_state')
            self._state = self._input_state
            if 'gray' in state_processor_params and state_processor_params['gray']:
                self._state = tf.image.rgb_to_grayscale(self._state)
                self._gray = True
            else:
                self._gray = False
            if 'crop_box' in state_processor_params:
                y1, x1, y2, x2 = state_processor_params['crop_box']
                self._state = tf.image.crop_to_bounding_box(self._state, y1, x1, y2, x2)
            if 'resize_shape' in state_processor_params:
                h, w = state_processor_params['resize_shape']
                self._state = tf.image.resize_images(self._state, [h, w],
                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            if 'frames_num' in state_processor_params:
                self._frames_num = state_processor_params['frames_num']
            else:
                self._frames_num = 1

    def _state_processor(self, state):
        return self.sess.run(self._state, feed_dict={self._input_state: state})

    @property
    def n_actions(self):
        return self._env.action_space.n

    @property
    def valid_actions(self):
        return self._valid_actions


class SimpleEnvironment:
    def __init__(self, config):
        self._env = gym.envs.make(config.env_name)
        self._valid_actions = [x for x in range(self.n_actions)]
        self._monitor_path = os.path.join(config.experiment_dir, "monitor/")
        self._env = wrappers.Monitor(self._env, self._monitor_path, resume=True,
                                     video_callable=lambda count: count % config.record_video_every == 0)

    def reset(self):
        state = self._env.reset()
        return state

    def step(self, action):
        next_state, reward, done, _ = self._env.step(action)
        return next_state, reward, done

    def sample_action(self):
        return np.random.choice(self._env.action_space.n)

    def submit(self, api_key):
        gym.upload(self._monitor_path, api_key=api_key)

    @property
    def n_actions(self):
        return self._env.action_space.n

    @property
    def valid_actions(self):
        return self._valid_actions

    @property
    def state_shape(self):
        return self._env.observation_space.shape
