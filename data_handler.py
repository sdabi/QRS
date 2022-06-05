import pandas as pd
import numpy as np
import defines
import random
from random_interaction_generator import random_interactions_data_generator

class Data_Handler():

    def __init__(self, random_data):
        if random_data == 0:
            self.orig_rating_df = pd.read_csv("ratings.csv")
            self.rating_df = self.orig_rating_df.copy()
            self.rating_df.loc[:,"rating"] = 1
            self.user_ids_list = np.sort(self.rating_df["userId"].unique().tolist())
            self.movie_ids_list = np.sort(self.rating_df["movieId"].unique().tolist())

        else:
            rig = random_interactions_data_generator(int((defines._NUM_OF_USERS*defines._NUM_OF_ITEMS)/10))
            self.rating_df = rig.get_rating_df()
            self.user_ids_list = list(range(defines._NUM_OF_USERS))
            self.movie_ids_list = list(range(defines._NUM_OF_ITEMS))


        self.user2user_encoded = {x: i for i, x in enumerate(self.user_ids_list)}
        self.user_encoded2user = {i: x for i, x in enumerate(self.user_ids_list)}

        self.movie2movie_encoded = {x: i for i, x in enumerate(self.movie_ids_list)}
        self.movie_encoded2movie = {i: x for i, x in enumerate(self.movie_ids_list)}

    def get_uninteracted_movieId_to_user(self, user_id):
        all_movies = set(self.movie_ids_list)
        interacted_movies = set(self.rating_df.loc[self.rating_df.userId == user_id, "movieId"].unique())
        return np.array(list(all_movies - interacted_movies))

    # input: user id - to whom his latest interaction will be marked as _REMOVED_INTER
    # output: movieId_removed - the movieId that removed for this user
    def remove_last_interaction_for_user(self, user_id):
        latest_inter_time = (max(self.rating_df[(self.rating_df.userId == user_id) & (self.rating_df.rating == 1)]['timestamp'].values))
        self.rating_df.loc[(self.rating_df.userId == user_id) & (
                self.rating_df.timestamp == latest_inter_time), 'rating'] = defines._REMOVED_INTER
        movieId_removed = self.rating_df.loc[
            (self.rating_df.userId == user_id) & (self.rating_df.timestamp == latest_inter_time), 'movieId'].values[0]
        return (user_id, movieId_removed)

    # remove the last interaction for every user
    # input: none
    # output: list of the removed movieIds (movieId in i'th pos - removed from i'th user)
    def remove_last_interaction_for_every_user(self):
        removed_movies = []
        for user_id in (self.rating_df["userId"].unique().tolist()):
            removed_movies.append(self.remove_last_interaction_for_user(user_id))
        return removed_movies


    # add bad sample to user - choosing 1 uninteracted item - and add it to the rating df with rating -1
    # input: user id
    # output: none (editing the rating_df on place)
    def bad_sample_to_user(self, user_id):
        # getting  list of uninteracted movies by user - list is the same size of the interacted list
        uninteracted_movieId = self.get_uninteracted_movieId_to_user(user_id)

        # calc the num of bad samples to add - the max number is the one that cause every item interaction
        num_of_interactions_for_user = len(list(self.rating_df.loc[(self.rating_df.userId == user_id) &
                                                                   (self.rating_df.rating == 1), "movieId"].values))

        num_of_bad_samples = min([num_of_interactions_for_user, random.randint(0,int(len(uninteracted_movieId)/2))])

        # sample movies to add bad sample to
        uninteracted_movieId_sampled = np.random.choice(uninteracted_movieId, num_of_bad_samples, replace=False)

        # building list of lists, each contains a movie not interacted - and then crating df
        un_inter_list = [[user_id, movieId, defines._BAD_SAMPLED_INTER, -1] for movieId in uninteracted_movieId_sampled]
        un_inter_df = pd.DataFrame(un_inter_list, columns=['userId', 'movieId', 'rating', 'timestamp'])

        # concating the bad sample df with the orig df
        self.rating_df = pd.concat([self.rating_df, un_inter_df], ignore_index=True)

    # for every user - add single bad sample
    def add_bad_sample_for_every_user(self):
        for user_id in (self.rating_df["userId"].unique().tolist()):
            self.bad_sample_to_user(user_id)

    # convert rating_df into interaction matrix - columns = items | rows = users
    # input: none
    # output: interaction matrix
    def get_interaction_table(self):
        R_df = self.rating_df.copy()

        for item in self.movie_ids_list:
            if len(R_df[R_df.movieId == item].values.tolist()) == 0:
                inter_as_df = pd.DataFrame([[0, item, 0, -1]], columns=['userId', 'movieId', 'rating', 'timestamp'])
                R_df = pd.concat([R_df, inter_as_df])

        print("num of users:", len(R_df["userId"].unique().tolist()))
        print("num of movies:", len(R_df["movieId"].unique().tolist()))


        R_df["user"] = R_df["userId"].map(self.user2user_encoded)
        R_df["movie"] = R_df["movieId"].map(self.movie2movie_encoded)
        R_df = R_df.drop(['userId', 'movieId', 'timestamp'], axis=1)
        R_df = R_df.pivot(index='user', columns='movie', values='rating').fillna(0)
        R_df = R_df.astype(int)
        R_df = R_df.reset_index(drop=True)
        R_df.columns = list(range(len(R_df.columns)))
        return R_df

    # return the movieId index in the interaction table
    # input: movieId
    # output: encoded movie
    def convert_movieId_to_movie_encode(self, movieID):
        return self.movie2movie_encoded[movieID]

    # return the userId index in the interaction table
    # input: userId
    # output: encoded user
    def convert_userId_to_user_encode(self, userID):
        return self.user2user_encoded[userID]


    def duplicated_user_inter(self, orig_user, new_user, shuffle_timestamp = 0):
        # delete the "new_user" entries from the existing DF
        self.rating_df = self.rating_df.drop(self.rating_df.index[self.rating_df['userId'] == new_user].tolist())

        # copy the "orig_user" entries to the DF"
        t = self.rating_df[self.rating_df['userId'] == orig_user].copy()
        t.loc[:, "userId"] = new_user

        if shuffle_timestamp:
            timestamp_arr = np.arange(t.shape[0])
            np.random.shuffle(timestamp_arr)
            t.loc[:, "timestamp"] = timestamp_arr # reshuffling the timestamp entry

        self.rating_df = pd.concat([self.rating_df, t])

        # restarting the indices
        self.rating_df = self.rating_df.reset_index(drop=True)


