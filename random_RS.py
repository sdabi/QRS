import numpy as np

class random_RS():

    def get_recommendation(self, user_index, movie_indexs, removed_movie):
        return np.random.rand(len(movie_indexs))
