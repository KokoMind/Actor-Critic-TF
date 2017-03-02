import tensorflow as tf


class Actor():
    """The actor class"""

    def __init__(self, sess, config):
        self._sess = sess

    def predict(self):
        pass

    def update(self):
        pass


class Critic():
    """The critic class"""

    def __init__(self, sess, config):
        self._sess = sess
        self._config = config
        self._name = config.critic_name
        self._build_model()

    def _build_model(self):
        with tf.variable_scope(self._name):
            self._state = None
            self._target = None

            self._out = tf.layers.Dense()

            self._value_estimate = tf.squeeze(self._out)
            self._loss = tf.squared_difference(self._out, self._target)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._config.learning_rate)
            self._update_step = self._optimizer.minimize(self._loss)


    def predict(self, state):
        return self._sess.run(self._value_estimate, feed_dict={self._state: state})

    def update(self, state, target):
        _, loss = self._sess.run([self._update_step, self._loss], feed_dict={self._state: state, self._target: target})
        return loss
