import numpy as np
import tensorflow as tf

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

class BinaryMF:
    def __init__(self, R, k, n_epoch, lr=0.03, l2=0.0001, seed=777):
        self.R = tf.convert_to_tensor(R, dtype=tf.float32)
        uninter_values = tf.convert_to_tensor([0, -2], dtype=tf.float32)
        self.mask = tf.not_equal(self.R, 0)
        self.m, self.n = R.shape
        self.k = k
        self.lr = lr
        self.l2 = l2
        self.tol = .001
        self.n_epoch = n_epoch
        # Initialize trainable weights.
        self.weight_init = tf.random_normal_initializer(seed=seed)
        self.P = tf.Variable(self.weight_init((self.m, self.k)))
        self.Q = tf.Variable(self.weight_init((self.n, self.k)))
        # Cast 1/-1 as binary encoding of 0/1.
        self.labels = tf.cast(tf.not_equal(tf.boolean_mask(self.R, self.mask), -1), dtype=tf.float32)
        self.predictions = 0

    def grad_update(self):
        with tf.GradientTape() as t:
            t.watch([self.P, self.Q])
            self.current_loss = self.loss()
        gP, gQ = t.gradient(self.current_loss, [self.P, self.Q])
        self.P.assign_sub(self.lr * gP)
        self.Q.assign_sub(self.lr * gQ)


    def train(self):
        for epoch in range(self.n_epoch):
            self.grad_update()
            if (epoch % 500 == 0):
                print("Iteration:", epoch, "; error:", self.current_loss.numpy())

    # The implementation is far from optimized since we don't need the product of entire P'Q.
    # We only need scores for non-missing entries.
    # The code is hence for educational purpose only.
    def loss(self):
        """Cross entropy loss."""
        logits = tf.boolean_mask(tf.matmul(self.P, self.Q, transpose_b=True), self.mask)
        logloss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits)
        mlogloss = tf.reduce_mean(logloss)
        l2_norm = tf.reduce_sum(self.P ** 2) + tf.reduce_sum(self.Q ** 2)
        return mlogloss + self.l2 * l2_norm

    def update_predictions(self):
        self.predictions = tf.sigmoid(tf.matmul(self.P, self.Q, transpose_b=True)).numpy()
        self.predictions = np.round(self.predictions, 2)
        #
        # b_mask = np.zeros_like(R)
        # b_mask[R.nonzero()] = 1
        # print(b_mask)
        # print(np.round(self.predictions * b_mask, 2))  # Check prediction on training entries.

    def full_matrix(self):
        return self.predictions

    def get_user_embedded_vectors(self):
        return self.P.numpy()

    def get_item_embedded_vectors(self):
        return self.Q.numpy()

    def get_recommendation(self, user_index, movie_indexs, removed_movie):
        return np.array([self.predictions[user_index][j] for j in movie_indexs])

