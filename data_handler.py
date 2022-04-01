import pandas as pd
import numpy as np
import defines
import random

class Data_Handler():

    def __init__(self, random_data):
        if (random_data == 0):
            self.orig_rating_df = pd.read_csv("ratings.csv")
            self.rating_df = self.orig_rating_df.copy()
            self.rating_df.loc[:,"rating"] = 1
            self.movies_df = pd.read_csv("movies.csv")
            self.user_ids_list = self.rating_df["userId"].unique().tolist()
            self.movie_ids_list = self.rating_df["movieId"].unique().tolist()
        else:
            self.rating_df = self.generate_random_interactions(int(defines._NUM_OF_USERS/2), defines._NUM_OF_USERS, defines._NUM_OF_ITEMS)
            self.user_ids_list = list(range(defines._NUM_OF_USERS))
            self.movie_ids_list = list(range(defines._NUM_OF_ITEMS))
            print(self.rating_df)


        self.user2user_encoded = {x: i for i, x in enumerate(self.user_ids_list)}
        self.user_encoded2user = {i: x for i, x in enumerate(self.user_ids_list)}



        self.movie2movie_encoded = {x: i for i, x in enumerate(self.movie_ids_list)}
        self.movie_encoded2movie = {i: x for i, x in enumerate(self.movie_ids_list)}

        print("num of users:", len(self.rating_df["userId"].unique().tolist()))
        print("num of movies:", len(self.rating_df["movieId"].unique().tolist()))


    def get_uninteracted_movieId_to_user(self, user_id):
        all_movies = set(self.movie_ids_list)
        interacted_movies = set(self.rating_df.loc[self.rating_df.userId == user_id, "movieId"].unique())
        return np.array(list(all_movies - interacted_movies))

    # input: user id - to whom his latest interaction will be marked as _REMOVED_INTER
    # output: movieId_removed - the movieId that removed for this user
    def remove_last_interaction_for_uesr(self, user_id):
        latest_inter_time = (max(self.rating_df[self.rating_df.userId == user_id]['timestamp'].values))
        self.rating_df.loc[(self.rating_df.userId == user_id) & (
                self.rating_df.timestamp == latest_inter_time), 'rating'] = defines._REMOVED_INTER
        movieId_removed = self.rating_df.loc[
            (self.rating_df.userId == user_id) & (self.rating_df.timestamp == latest_inter_time), 'movieId'].values[0]
        return movieId_removed

    # remove the last interaction for every user
    # input: none
    # output: list of the removed movieIds (movieId in i'th pos - removed from i'th user)
    def remove_last_interaction_for_every_user(self):
        removed_movies = []
        for user_id in (self.rating_df["userId"].unique().tolist()):
            removed_movies.append(self.remove_last_interaction_for_uesr(user_id))
        return removed_movies


    # add bad sample to user - choosing 1 uninteracted item - and add it to the rating df with rating -1
    # input: user id
    # output: none (editing the rating_df on place)
    def bad_sample_to_user(self, user_id):
        # getting  list of uninteracted movies by user - list is the same size of the interacted list
        uninteracted_movieId = self.get_uninteracted_movieId_to_user(user_id)
        num_of_interactions_for_user = len(list(self.rating_df.loc[(self.rating_df.userId == user_id) &
                                                                   (self.rating_df.rating == 1), "movieId"].values))
        uninteracted_movieId_sampled = np.random.choice(uninteracted_movieId, num_of_interactions_for_user,
                                                        replace=False)

        # building list of lists, each contains a movie not interacted - and then crating df
        un_inter_list = [[user_id, movieId, defines._BAD_SAMPLED_INTER, 0] for movieId in uninteracted_movieId_sampled]
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
        R_df["user"] = R_df["userId"].map(self.user2user_encoded)
        R_df["movie"] = R_df["movieId"].map(self.movie2movie_encoded)
        R_df = R_df.drop(['userId', 'movieId', 'timestamp'], axis=1)
        R_df = R_df.pivot(index='user', columns='movie', values='rating').fillna(0)
        R_df = R_df.astype(int)
        R_df = R_df.reset_index(drop=True)
        R_df.columns = list(range(len(R_df.columns)))

        # add missing columns - items without any interaction
        all_items_columns = list(range(len(self.movie_ids_list)))
        R_df = R_df.reindex(columns=all_items_columns, fill_value=0)
        return R_df

    # return the movieId index in the interaction table
    # input: movieId
    # output: encoded movie
    def convert_movieId_to_movie_encode(self, movieID):
        return self.movie2movie_encoded[movieID]


    def generate_random_interactions(self, num_of_uniq_users, num_of_users, num_of_items):
        inter_list = []

        # randomizing unique users
        for user in range(num_of_uniq_users):
            num_of_interactions_for_user = random.randint(int(num_of_items/6), int(num_of_items/4))
            inter_for_user = np.random.choice(np.arange(0, num_of_items), replace=False, size= num_of_interactions_for_user)
            print('a',inter_for_user)
            # inter_for_user = random.sample(range(num_of_items), num_of_interactions_for_user)
            for item, timestep in enumerate(inter_for_user):
                inter_list.append([user, item, 1, random.randint(1, 100)])

        # copy not-unique users
        un_inter_df = pd.DataFrame(inter_list, columns=['userId', 'movieId', 'rating', 'timestamp'])
        for user in range(num_of_uniq_users,num_of_users):
            uniq_user_sampled = random.randint(0, num_of_uniq_users-1)
            df_to_append = un_inter_df.loc[un_inter_df.userId == uniq_user_sampled, :].copy()
            df_to_append.loc[:, 'userId'] = user
            un_inter_df = pd.concat([un_inter_df, df_to_append])
        return un_inter_df



