import numpy as np

class MF_ALS():

    def __init__(self, R, K, lambda_, iterations):
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
        self.P = 5 * np.random.rand(self.num_users, self.K)
        self.Q = 5 * np.random.rand(self.K, self.num_items)
        self.lambda_ = lambda_
        self.iterations = iterations

    def get_error(self):
        # This calculates the MSE of nonzero elements
        return np.sum((self.R * (self.R - np.dot(self.P, self.Q))) ** 2) / np.sum(self.R)


    def runALS(self):

        MSE_List = []

        print ("Starting Iterations")
        for iter in range(self.iterations):
            for i, Ri in enumerate(self.R): # i = user, Ri is a vector contains all the interactions for user i
                self.P[i] = np.linalg.solve(np.dot(self.Q, np.dot(np.diag(Ri), self.Q.T)) + self.lambda_ * np.eye(self.K),
                                           np.dot(self.Q, np.dot(np.diag(Ri), self.R[i].T))).T
            # print ("Error after solving for User Matrix:", self.get_error())

            for j, Rj in enumerate(self.R.T):
                self.Q[:,j] = np.linalg.solve(np.dot(self.P.T, np.dot(np.diag(Rj), self.P)) + self.lambda_ * np.eye(self.K),
                                         np.dot(self.P.T, np.dot(np.diag(Rj), self.R[:, j])))
            # print ("Error after solving for Item Matrix:", self.get_error())

            MSE_List.append(self.get_error())
            # print ('%sth iteration is complete...' % iter)


    def get_recommendation(self, user_index, movie_indexs):
        return np.array([np.dot(self.P[user_index] ,self.Q[: ,j]) for j in movie_indexs])

    def get_embedded_vectors(self):
        return (self.P, self.Q.T)

    def full_matrix(self):
        return self.P.dot(self.Q)
