
import pandas as pd
import numpy as np
import defines
import random

class random_interactions_data_generator():

    def __init__(self, num_of_interactions, num_of_inter_per_user, num_of_bad_inters):

        all_interacted_movies_per_user = {}
        for user in range(defines._NUM_OF_USERS):
            all_interacted_movies_per_user[user] = []

        self.rating_df = pd.DataFrame([], columns=['userId', 'movieId', 'rating', 'timestamp'])

        # TO ENABLE RANDOM NUM OF INTERACTION FOR USER:
        # comment the following 2 for loops - and uncomment the 3'rd one + uncomment user_to_add_inter
        next_top_index_in_df = 0
        timestamp = 0

        # -------------------- ADDING POSITIVE INTERACTIONS -----------------------
        for inter in range(num_of_inter_per_user):
            for user_to_add_inter in range(defines._NUM_OF_USERS):
                inter_counts = np.zeros(defines._NUM_OF_ITEMS)

                # searching for users with same interactions
                for inter in all_interacted_movies_per_user[user_to_add_inter]:
                    users_with_same_inter = self.rating_df.loc[self.rating_df.movieId == inter, 'userId'].values.tolist()
                    users_with_same_inter.remove(user_to_add_inter)

                    # for each user with same interaction - get all his other interactions
                    for user_with_same_inter in users_with_same_inter:
                        # and sum them into array (histograms)
                        inter_counts[all_interacted_movies_per_user[user_with_same_inter]] += 1

                # removing from the histogram all the previous interaction for the user
                inter_counts[all_interacted_movies_per_user[user_to_add_inter]] = -1

                max_interacted_item = np.random.choice(np.where(inter_counts == inter_counts.max())[0], replace=False, size=1)[0]
                self.rating_df.loc[next_top_index_in_df] = [user_to_add_inter, max_interacted_item, 1, timestamp]
                all_interacted_movies_per_user[user_to_add_inter].append(max_interacted_item)
                next_top_index_in_df += 1
                timestamp += 1

        # make sure that for every user there are at least 2 interactions
        # for i in range(2):
            # for user_to_add_inter in range(defines._NUM_OF_USERS):
            #     inter_counts = np.zeros(defines._NUM_OF_ITEMS)
            #
            #     # getting all his previous interactions
            #     all_interacted_movies_for_user = self.rating_df.loc[
            #         self.rating_df.userId == user_to_add_inter, 'movieId'].values.tolist()
            #
            #     # searching for users with same interactions
            #     for inter in all_interacted_movies_for_user:
            #         users_with_same_inter = self.rating_df.loc[self.rating_df.movieId == inter, 'userId'].values.tolist()
            #
            #         # for each user with same interaction - get all his other interactions
            #         for user_with_same_inter in users_with_same_inter:
            #             movies_user_with_same_inter_also_had = self.rating_df.loc[
            #                 self.rating_df.userId == user_with_same_inter, 'movieId'].values.tolist()
            #
            #             # and sum them into array (histograms)
            #             inter_counts[movies_user_with_same_inter_also_had] += 1
            #
            #     # removing from the histogram all the previous interaction for the user
            #     inter_counts[all_interacted_movies_for_user] = -1
            #
            #     max_interacted_item = np.random.choice(np.where(inter_counts == inter_counts.max())[0], replace=False, size=1)[
            #         0]
            #     # print("user",user_to_add_inter,"choose",max_interacted_item,"score",inter_counts.max())
            #     inter_as_df = pd.DataFrame([[user_to_add_inter, max_interacted_item, 1, timestamp]],
            #                                columns=['userId', 'movieId', 'rating', 'timestamp'])
            #     self.rating_df = pd.concat([self.rating_df, inter_as_df])
            #     timestamp +=1

        # add bad interactions - based on interaction which user loved/not loved
        # for inter_num in range(num_of_bad_inters):
        # for inter in range(num_of_inter_per_user -1 ):
        #     for user in range(defines._NUM_OF_USERS):
        #         inter_counts = np.zeros(defines._NUM_OF_ITEMS)
        #         # picking a user to add interaction to
        #         user_to_add_inter = user
        #         #user_to_add_inter = random.randint(0, defines._NUM_OF_USERS-1)
        #
        #         # getting all his previous interactions (both good and bad)
        #         all_interacted_movies_for_user = self.rating_df.loc[(self.rating_df.userId == user_to_add_inter), 'movieId'].values.tolist()
        #
        #         # searching for users with same interactions (both good and bad)
        #         for inter in all_interacted_movies_for_user:
        #             # getting a user that had same interaction with same movie, both need to love or not love the same movie
        #             user_rating = self.rating_df.loc[(self.rating_df.userId == user_to_add_inter) & (self.rating_df.movieId == inter), 'rating'].values.tolist()[0]
        #             users_with_same_inter = self.rating_df.loc[(self.rating_df.movieId == inter) & (self.rating_df.rating == user_rating), 'userId'].values.tolist()
        #
        #             # for each user with same interaction - get all his other bad interactions
        #             for user_with_same_inter in users_with_same_inter:
        #                 movies_user_with_same_inter_also_had = self.rating_df.loc[(self.rating_df.userId == user_with_same_inter) & (self.rating_df.rating == defines._BAD_SAMPLED_INTER), 'movieId'].values.tolist()
        #
        #                 # and sum them into array (histograms)
        #                 inter_counts[movies_user_with_same_inter_also_had] += 1
        #
        #         # removing from the histogram all the previous interaction for the user
        #         inter_counts[all_interacted_movies_for_user] = -1
        #
        #         max_interacted_item = np.random.choice(np.where(inter_counts == inter_counts.max())[0], replace=False, size=1)[0]
        #         # print("user",user_to_add_inter,"choose",max_interacted_item,"score",inter_counts.max())
        #         inter_as_df = pd.DataFrame([[user_to_add_inter, max_interacted_item, defines._BAD_SAMPLED_INTER, inter_num]], columns=['userId', 'movieId', 'rating', 'timestamp'])
        #         self.rating_df = pd.concat([self.rating_df, inter_as_df])

        # -------------------- ADDING NEGATIVE INTERACTIONS -----------------------
        all_items_set = set(list(range(defines._NUM_OF_ITEMS)))
        for user_to_add_inter in range(defines._NUM_OF_USERS):

            uninteracted_items_list = np.array(list(all_items_set - set(all_interacted_movies_per_user[user_to_add_inter])))
            negative_sampled_items = np.random.choice(uninteracted_items_list, replace=False, size=num_of_inter_per_user-1)

            for negative_item in negative_sampled_items:
                self.rating_df.loc[next_top_index_in_df] = [user_to_add_inter, negative_item, -1, timestamp]
                all_interacted_movies_per_user[user_to_add_inter].append(negative_item)
                next_top_index_in_df += 1
                timestamp += 1



        self.rating_df = self.rating_df.reset_index(drop=True)



    def get_rating_df(self):
        return self.rating_df
