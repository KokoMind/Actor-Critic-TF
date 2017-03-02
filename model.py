import tensorflow as tf


class Actor():
    """The actor class"""

    def __init__(self, sess, config, num_actions, observation_shape):
        self._sess = sess

        self._state = tf.placeholder(dtype=tf.float32, shape=observation_shape, name='state')
        self._action = tf.placeholder(dtype=tf.int32, shape=[1], name='action')
        self._target = tf.placeholder(dtype=tf.float32, shape=[1], name='target')

        self._output_layer = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(self._state, 0), num_outputs=num_actions,
                                                               activation_fn=None, weights_initializer=tf.zeros_initializer)
        self._action_probs = tf.squeeze(tf.nn.softmax(self._output_layer))
        self._picked_action_prob = tf.gather(self._action_probs, self._action)

        self._loss = -tf.log(self._picked_action_prob) * self._target

        self._optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        self._train_op = self.optimizer.minimize(self._lose)

    def predict(self, s):
        return self._sess.run(self._action_probs, {self._state: s})

    def update(self, s, a, target):
        self._sess.run(self._train_op, {self._state: s, self._action: a, self._target: target})


class Critic():
    """The critic class"""

    def __init__(self, sess, config):
        self._sess = sess

    def predict(self):
        pass

    def update(self):
        pass
