import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


class MF():

    def __init__(self, R, K, alpha, beta, norm_weight, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.norm_weight = norm_weight
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if (self.R[i, j] == 1 or self.R[i, j] == -1)
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            # if (i + 1) % 10 == 0:
            #     print("Iteration: %d ; error = %.4f" % (i + 1, mse))

        return training_process


    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            norm = self.get_norm(i, j)
            e = (r - prediction) + (self.norm_weight * norm)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])

        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            norm = self.get_norm(i, j)
            e = (r - prediction) + (self.norm_weight * norm)

            # Update biases
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])


    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        #         e = self.R - predicted
        #         e = e**2
        #         error = np.sum(e)
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def get_norm(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        norm = self.b_u[i] ** 2 + self.b_i[j] ** 2 + self.P[i, :].dot(self.P[i, :].T) + self.Q[j, :].dot(self.Q[j, :].T)
        return norm

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)

    def get_embedded_vectors(self):
        return (self.P, self.Q)

    def get_recommendation(self, user_index, movie_indexs):
        return np.array([self.get_rating(user_index, j) for j in movie_indexs])
