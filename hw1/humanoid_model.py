import numpy as np
import tensorflow as tf
from model import Model

class HumanoidModel(Model):

    def set_data(self, x, y):

        self.observations = x
        self.actions = y

        self.data_size = self.observations.shape[0]

    def build(self):

        OBS_SIZE = self.observations.shape[-1]
        ACTION_SIZE = self.actions.shape[-1]

        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        regularizer = tf.contrib.layers.l2_regularizer(0.1, scope=None)

        self.input = tf.placeholder(dtype=tf.float32, shape=(None, OBS_SIZE))

        hidden = tf.layers.dense(self.input, ACTION_SIZE, kernel_initializer=initializer,
                                 kernel_regularizer=regularizer, activation=tf.tanh)

        hidden = tf.layers.dense(hidden, ACTION_SIZE, kernel_initializer=initializer,
                        kernel_regularizer=regularizer, activation=tf.tanh)

        self.output = tf.layers.dense(hidden, ACTION_SIZE, kernel_initializer=initializer,
                                      kernel_regularizer=regularizer, activation=tf.tanh)

        self.label = tf.placeholder(dtype=tf.float32, shape=(None, ACTION_SIZE))

        # loss function is L2 loss
        self.loss = 0.5 * tf.reduce_mean(tf.square(self.label - self.output))

        # optimization
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-08)

        self.train_step = self.optimizer.minimize(loss=self.loss)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def train(self, n_epochs=2000):

        batch_size = self.batch_size
        n_batches = int(self.data_size / batch_size)
        writer = tf.summary.FileWriter("log/...", self.sess.graph)

        for e in range(n_epochs):
            print(e)
            avg_loss = 0
            for i in range(n_batches):
                obs = self.observations[i * batch_size:(i + 1) * batch_size, :]
                act = self.actions[i * batch_size:(i + 1) * batch_size]

                loss_value, _ = self.sess.run([self.loss, self.train_step], feed_dict={self.input: obs, self.label: act})
                avg_loss += loss_value / n_batches

            print(avg_loss)

        writer.close()

        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "humanoid_model.ckpt")
        print("Model saved in path: %s" % save_path)

    def predict(self, obs):

        return self.sess.run(self.output, feed_dict={self.input:obs})

    def load(self, filename):
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)


