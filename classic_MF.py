import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



class MF():

    def __init__(self, R, K, alpha, beta, iterations):
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
            total_error = self.sgd()
            training_process.append(total_error)
            if (i + 1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i + 1, total_error))
        print("training Done")
        return training_process


    def sgd(self):

        total_e = 0
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            prediction_after_sig = (sigmoid(prediction) - 0.5) * 2

            e = (r - prediction_after_sig)
            total_e += e**2 + self.beta * self.get_norm(i, j)

            delta_prediction = - 2 * np.exp(-prediction) / ((1 + np.exp(-prediction))**2)
            delta_prediction = 2 * e * delta_prediction

            self.P[i, :] -= self.alpha * ((delta_prediction * self.Q[j, :]) + (2 * self.beta * self.P[i, :]))
            self.Q[j, :] -= self.alpha * ((delta_prediction * self.P[i, :]) + (2 * self.beta * self.Q[j, :]))

            self.b_u[i] -= self.alpha * ((delta_prediction) + (2 * self.beta * self.b_u[i]))
            self.b_i[j] -= self.alpha * ((delta_prediction) + (2 * self.beta * self.b_i[j]))

            #
            #
            # e = (r - prediction_after_sig)
            # total_e += e**2 + self.beta * self.get_norm(i, j)
            # self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            # self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])
            #
            # # Update biases
            # self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            # self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
        return total_e

    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        # if abs(prediction) > 100:
        #     prediction = np.sign(prediction)*100
        # return (sigmoid(prediction)-0.5)*2
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
        return (sigmoid(self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T))-0.5)*2

    def get_user_embedded_vectors(self):
        return (self.P)


    def get_item_embedded_vectors(self):
        return (self.Q)

    def get_recommendation(self, user_index, movie_indexs, removed_movie):
        return np.array([self.get_rating(user_index, j) for j in movie_indexs])
