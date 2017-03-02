import tensorflow as tf
import numpy as np
from config import get_config


class Actor:
    """The actor class"""

    def __init__(self, sess, num_actions, observation_shape, config):
        self._sess = sess

        self._state = tf.placeholder(dtype=tf.float32, shape=observation_shape, name='state')
        self._action = tf.placeholder(dtype=tf.int32, name='action')
        self._target = tf.placeholder(dtype=tf.float32, name='target')

        self._hidden_layer = tf.layers.dense(inputs=tf.expand_dims(self._state, 0), units=32, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
        self._output_layer = tf.layers.dense(inputs=self._hidden_layer, units=num_actions, kernel_initializer=tf.zeros_initializer())
        self._action_probs = tf.squeeze(tf.nn.softmax(self._output_layer))
        self._picked_action_prob = tf.gather(self._action_probs, self._action)

        self._loss = -tf.log(self._picked_action_prob) * self._target

        self._optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        self._train_op = self._optimizer.minimize(self._loss)

    def predict(self, s):
        return self._sess.run(self._action_probs, {self._state: s})

    def update(self, s, a, target):
        self._sess.run(self._train_op, {self._state: s, self._action: a, self._target: target})


class Critic:
    """The critic class"""

    def __init__(self, sess, observation_shape, config):
        self._sess = sess
        self._config = config
        self._name = config.critic_name
        self._observation_shape = observation_shape
        self._build_model()

    def _build_model(self):
        with tf.variable_scope(self._name):
            self._state = tf.placeholder(dtype=tf.float32, shape=self._observation_shape, name='state')
            self._target = tf.placeholder(dtype=tf.float32, name='target')

            self._hidden_layer = tf.layers.dense(inputs=tf.expand_dims(self._state, 0), units=32, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
            self._out = tf.layers.dense(inputs=self._hidden_layer, units=1, kernel_initializer=tf.zeros_initializer())

            self._value_estimate = tf.squeeze(self._out)
            self._loss = tf.squared_difference(self._out, self._target)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._config.learning_rate)
            self._update_step = self._optimizer.minimize(self._loss)

    def predict(self, s):
        return self._sess.run(self._value_estimate, feed_dict={self._state: s})

    def update(self, s, target):
        self._sess.run(self._update_step, feed_dict={self._state: s, self._target: target})


def test():
    """Function to test the model Implementation"""
    state = np.random.rand(4)
    target = np.random.rand(1)
    action = 1

    tf.reset_default_graph()
    sess = tf.Session()

    config = get_config()

    actor = Actor(sess, 4, [4], config)
    critic = Critic(sess, [4], config)

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    actor.predict(state)
    critic.predict(state)
    actor.update(state, action, target)
    critic.update(state, target)
    pass


if __name__ == "__main__":
    test()
