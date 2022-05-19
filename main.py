import numpy as np
import matplotlib.pyplot as plt

import defines
from classic_MF import MF
from data_handler import Data_Handler
from embedded_QRS import embedded_QRS
from random_RS import random_RS
import visualiser

# exporting the recommendation_sets and the R_df
def export_data():
    out_recommendation_sets = np.empty(len(recommendation_sets), dtype=object)
    out_recommendation_sets[:] = recommendation_sets
    with open('recommendation_sets.npy', 'wb') as f:
        np.save(f, out_recommendation_sets)
    with open('R_df_as_numpy.npy', 'wb') as f:
        np.save(f, R_df_as_numpy)

# importing the recommendation_sets and the R_df - returned as np
def load_data():
    with open('recommendation_sets.npy', 'rb') as f:
        out_recommendation_sets = np.load(f, allow_pickle=True)
    with open('R_df_as_numpy.npy', 'rb') as f:
        R_df_as_numpy = np.load(f, allow_pickle=True)
    return (out_recommendation_sets, R_df_as_numpy)


# input: reco_hit_index arr - for every recommendation - where the LOO item was
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



def TEST_MODEL(recommendation_sets, MODEL):
    reco_hit_index = []
    for user, removed_movie, uninter_movies in recommendation_sets:
        # get recommendations from the model - on uninter_movies list
        reco_scores = MODEL.get_recommendation(user, uninter_movies)

        # getting the index of the LOO item from the recommendations
        reco_hit_index.append(get_LOO_index_in_scores_array(uninter_movies, removed_movie, reco_scores))

    reco_hit_index = np.array(reco_hit_index)
    HRK = create_HRK(reco_hit_index)

    return HRK




if __name__ == '__main__':
    # ----------------------------- DATA PREPARATION ------------------------------------
    dh = Data_Handler(random_data=1)
    dh.add_bad_sample_for_every_user()
    dh.duplicated_user_inter(0, 1)
    LOO = dh.remove_last_interaction_for_every_user()
    R_df = dh.get_interaction_table()
    recommendation_sets = create_recommendation_sets(LOO, dh)
    R_df_as_numpy = R_df.to_numpy()
    user_items_removed_indices = [(x[0], x[1]) for x in recommendation_sets]
    visualiser.print_colored_matrix(R_df.to_numpy(), [user_items_removed_indices])


    # -------------------------------- RANDOM RECOMMENDATION --------------------------------
    RAND_RECO = random_RS()
    HRK_RAND_RECO = TEST_MODEL(recommendation_sets, RAND_RECO)


    # -------------------------------- TRAINING CLASSIC MF --------------------------------
    SDG_MF = MF(R_df.to_numpy(), defines._EMBEDDING_SIZE, alpha=0.1, beta=0.01, norm_weight=0.001, iterations=2000)
    SDG_MF.train()
    HRK_MF = TEST_MODEL(recommendation_sets, SDG_MF)
    visualiser.print_colored_matrix(SDG_MF.full_matrix(), [user_items_removed_indices])
    visualiser.plot_HRK([HRK_MF, HRK_RAND_RECO], ["MF", "RAND_RECO"])


    # -------------------------------- TRAINING EMBEDDED QRS --------------------------------
    user_embedded_vecs = SDG_MF.get_user_embedded_vectors()
    item_embedded_vecs = SDG_MF.get_item_embedded_vectors()
    visualiser.plot_embedded_vecs(user_embedded_vecs)
    QRS = embedded_QRS(R_df.to_numpy(), user_embedded_vecs, item_embedded_vecs, train_steps=50)
    QRS.train()
    HRK_QRS = TEST_MODEL(recommendation_sets, QRS)

    visualiser.plot_HRK([HRK_MF, HRK_QRS, HRK_RAND_RECO], ["MF", "QRS", "RAND_RECO"])




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


