import pandas as pd
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt
import math

import random
import time
from threading import Thread, Lock
import queue

import defines
from classic_MF import MF
from classic_als_MF import MF_ALS
from data_handler import Data_Handler
from random_interaction_generator import random_interactions_data_generator


# input: reco_hit_index arr - for every recommandation - where the LOO item was
# output: HR@K arr
def create_HRK(hit_arr):
    hits_ind = np.zeros(11)
    for hit in hit_arr:
        if hit <= 10:
            hits_ind[hit] += 1
    _hit_arr = np.cumsum(hits_ind)
    _hit_arr = _hit_arr/len(hit_arr)
    return _hit_arr



# input:  uninteracted movies array
#         LOO item
#         scores array - in corelative order to uninteracted movies
# output: the index of the LOO item in the scores array
def get_LOO_index_in_scores_array(uninteracted_movies, LOO_item, reco_scores):
    t = np.array(uninteracted_movies)
    desired_inter_index_pos = np.where(t == LOO_item)[0][0]
    return (np.where(reco_scores.argsort()[::-1][:len(reco_scores)]==desired_inter_index_pos)[0])[0]+1


# input:    1. list of tuples
#               A. userID - who we removed the interaction from
#               B. moviedId - which was removed from the user
#           2. if num_of_uninter_per_user == 0 - than taking all uninteracted movies
# output:   1. list of triples:
#               A. user index (encoded)
#               B. the movie index which removed (encoded)
#               C. list of uninteracted movies - contains the removed interaction movie (encoded)
def create_recommendation_sets(LOO, dh, num_of_uninter_per_user = 0):
    recommendation_sets = []
    for user, movieId in LOO:
        # getting all uninteracted movies
        uninteracted_movies = dh.get_uninteracted_movieId_to_user(user)

        # sample X of the uninteracted movies
        if (num_of_uninter_per_user != 0):
            uninteracted_movies = np.random.choice(uninteracted_movies, num_of_uninter_per_user, replace=False)
        uninteracted_movies = [dh.convert_movieId_to_movie_encode(x) for x in uninteracted_movies]

        # convert the LOO movieId to encoded movie and add it to list
        LOO_encoded_movie = dh.convert_movieId_to_movie_encode(movieId)
        uninteracted_movies.append(LOO_encoded_movie)
        uninteracted_movies = list(set(uninteracted_movies))

        user_encoded = dh.convert_userId_to_user_encode(user)
        recommendation_sets.append((user_encoded, LOO_encoded_movie, uninteracted_movies))
    return recommendation_sets


def run_MF(recommendation_sets, SDG_MF):
    reco_hit_index = []
    for user, removed_movie, uninter_movies in recommendation_sets:
        # get MF recommendations
        reco_scores = SDG_MF.get_recommendation(user, uninter_movies)

        reco_hit_index.append(get_LOO_index_in_scores_array(uninter_movies, removed_movie, reco_scores))
        print("reco_hit_index", reco_hit_index[-1])

    reco_hit_index = np.array(reco_hit_index)
    HRK = create_HRK(reco_hit_index)
    return HRK



if __name__ == '__main__':
    # ----------------------------- DATA PREPARATION ------------------------------------
    dh = Data_Handler(random_data=1)
    LOO = dh.remove_last_interaction_for_every_user()
    dh.add_bad_sample_for_every_user()
    R_df = dh.get_interaction_table()
    recommendation_sets = create_recommendation_sets(LOO, dh)


    # -------------------------------- TRAINING CLASSIC MF --------------------------------
    SDG_MF = MF(R_df.to_numpy(), defines._EMBEDDING_SIZE, alpha=0.1, beta=0.01, norm_weight=0.001, iterations=100)
    SDG_MF.train()
    HRK_MF = run_MF(recommendation_sets, SDG_MF)


    plt.plot(HRK_MF,label ="hits_per_K_RANDOM",color='green')
    y_axis = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.ylim([0, 1])
    plt.yticks(y_axis)
    plt.xlim([1, len(HRK_MF)])
    plt.legend()
    plt.grid()
    plt.show()




# ======================================= FULL ORIG DATA TESTING =======================================
if __name__ == 'FULL_ORIG_DATA_TESTING':


    # ----------------------------- DATA PREPARATION ------------------------------------
    dh = Data_Handler(random_data=0)
    LOO = dh.remove_last_interaction_for_every_user()
    dh.add_bad_sample_for_every_user()
    R_df = dh.get_interaction_table()
    print(R_df)
    recommendation_sets = create_recommendation_sets(LOO, dh, 100)

    # -------------------------------- TRAINING CLASSIC MF --------------------------------
    SDG_MF = MF(R_df.to_numpy(), defines._EMBEDDING_SIZE, alpha=0.1, beta=0.01, norm_weight=0.001, iterations=50)
    SDG_MF.train()
    HRK_MF = run_MF(recommendation_sets, SDG_MF)


    plt.plot(HRK_MF,label ="hits_per_K_RANDOM",color='green')
    y_axis = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.ylim([0, 1])
    plt.yticks(y_axis)
    plt.xlim([1, len(HRK_MF)])
    plt.legend()
    plt.grid()
    plt.show()


