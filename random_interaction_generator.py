
import pandas as pd
import numpy as np
import defines
import random

class random_interactions_data_generator():

    def __init__(self, num_of_interactions, num_of_bad_inters):

        self.rating_df = pd.DataFrame([], columns=['userId', 'movieId', 'rating', 'timestamp'])

        for inter_num in range(num_of_interactions):
            inter_counts = np.zeros(defines._NUM_OF_ITEMS)
            # picking a user to add interaction to
            user_to_add_inter = random.randint(0, defines._NUM_OF_USERS-1)

            # getting all his previous interactions
            all_interacted_movies_for_user = self.rating_df.loc[self.rating_df.userId == user_to_add_inter, 'movieId'].values.tolist()

            # searching for users with same interactions
            for inter in all_interacted_movies_for_user:
                users_with_same_inter = self.rating_df.loc[self.rating_df.movieId == inter, 'userId'].values.tolist()

                # for each user with same interaction - get all his other interactions
                for user_with_same_inter in users_with_same_inter:
                    movies_user_with_same_inter_also_had = self.rating_df.loc[self.rating_df.userId == user_with_same_inter, 'movieId'].values.tolist()

                    # and sum them into array (histograms)
                    inter_counts[movies_user_with_same_inter_also_had] += 1

            # removing from the histogram all the previous interaction for the user
            inter_counts[all_interacted_movies_for_user] = -1

            max_interacted_item = np.random.choice(np.where(inter_counts == inter_counts.max())[0], replace=False, size=1)[0]
            # print("user",user_to_add_inter,"choose",max_interacted_item,"score",inter_counts.max())
            inter_as_df = pd.DataFrame([[user_to_add_inter, max_interacted_item, 1, inter_num]], columns=['userId', 'movieId', 'rating', 'timestamp'])
            self.rating_df = pd.concat([self.rating_df, inter_as_df])

        # make sure that for every user there are at least 2 interactions
        timestamp = num_of_interactions
        for i in range(2):
            for user_to_add_inter in range(defines._NUM_OF_USERS):
                inter_counts = np.zeros(defines._NUM_OF_ITEMS)

                # getting all his previous interactions
                all_interacted_movies_for_user = self.rating_df.loc[
                    self.rating_df.userId == user_to_add_inter, 'movieId'].values.tolist()

                # searching for users with same interactions
                for inter in all_interacted_movies_for_user:
                    users_with_same_inter = self.rating_df.loc[self.rating_df.movieId == inter, 'userId'].values.tolist()

                    # for each user with same interaction - get all his other interactions
                    for user_with_same_inter in users_with_same_inter:
                        movies_user_with_same_inter_also_had = self.rating_df.loc[
                            self.rating_df.userId == user_with_same_inter, 'movieId'].values.tolist()

                        # and sum them into array (histograms)
                        inter_counts[movies_user_with_same_inter_also_had] += 1

                # removing from the histogram all the previous interaction for the user
                inter_counts[all_interacted_movies_for_user] = -1

                max_interacted_item = np.random.choice(np.where(inter_counts == inter_counts.max())[0], replace=False, size=1)[
                    0]
                # print("user",user_to_add_inter,"choose",max_interacted_item,"score",inter_counts.max())
                inter_as_df = pd.DataFrame([[user_to_add_inter, max_interacted_item, 1, timestamp]],
                                           columns=['userId', 'movieId', 'rating', 'timestamp'])
                self.rating_df = pd.concat([self.rating_df, inter_as_df])
                timestamp +=1

        # add bad interactions - based on interaction which user loved/not loved
        for inter_num in range(num_of_bad_inters):
            inter_counts = np.zeros(defines._NUM_OF_ITEMS)
            # picking a user to add interaction to
            user_to_add_inter = random.randint(0, defines._NUM_OF_USERS-1)

            # getting all his previous interactions (both good and bad)
            all_interacted_movies_for_user = self.rating_df.loc[(self.rating_df.userId == user_to_add_inter), 'movieId'].values.tolist()

            # searching for users with same interactions (both good and bad)
            for inter in all_interacted_movies_for_user:
                # getting a user that had same interaction with same movie, both need to love or not love the same movie
                user_rating = self.rating_df.loc[(self.rating_df.userId == user_to_add_inter) & (self.rating_df.movieId == inter), 'rating'].values.tolist()[0]
                users_with_same_inter = self.rating_df.loc[(self.rating_df.movieId == inter) & (self.rating_df.rating == user_rating), 'userId'].values.tolist()

                # for each user with same interaction - get all his other bad interactions
                for user_with_same_inter in users_with_same_inter:
                    movies_user_with_same_inter_also_had = self.rating_df.loc[(self.rating_df.userId == user_with_same_inter) & (self.rating_df.rating == defines._BAD_SAMPLED_INTER), 'movieId'].values.tolist()

                    # and sum them into array (histograms)
                    inter_counts[movies_user_with_same_inter_also_had] += 1

            # removing from the histogram all the previous interaction for the user
            inter_counts[all_interacted_movies_for_user] = -1

            max_interacted_item = np.random.choice(np.where(inter_counts == inter_counts.max())[0], replace=False, size=1)[0]
            # print("user",user_to_add_inter,"choose",max_interacted_item,"score",inter_counts.max())
            inter_as_df = pd.DataFrame([[user_to_add_inter, max_interacted_item, defines._BAD_SAMPLED_INTER, inter_num]], columns=['userId', 'movieId', 'rating', 'timestamp'])
            self.rating_df = pd.concat([self.rating_df, inter_as_df])


        self.rating_df = self.rating_df.reset_index(drop=True)



    def get_rating_df(self):
        return self.rating_df
