import random

import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane import numpy as np
import time
import itertools

import math
import defines
import visualiser
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.templates.layers import BasicEntanglerLayers

n_item_wires = (int(math.log(defines._NUM_OF_ITEMS, 2)))
n_user_wires = 0
n_wires = n_user_wires + n_item_wires

item_wires = list(range(n_item_wires))
user_wires = list(range(n_item_wires, n_wires))
wires_list = item_wires + user_wires


# this cricuit is used to plot the embeding quantum state only
dev_embedded_ItemItem_embedding = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_embedded_ItemItem_embedding)
def embedded_QRS_circ_embedding(user_params):
    for wire in item_wires:
        qml.Hadamard(wire)

    for p in user_params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)

    return qml.probs(wires=item_wires)


# ============================ WITHOUT GROUPS ===================================
dev_embedded_ItemItem_reco_without_groups = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_embedded_ItemItem_reco_without_groups)
def embedded_QRS_circ_reco_without_groups(params, user_params):
    for wire in item_wires:
        qml.Hadamard(wire)

    for p in user_params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)

    for p in params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)

    return qml.probs(wires=item_wires)


# ============================ WITH GROUPS ===================================
dev_embedded_ItemItem_reco = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_embedded_ItemItem_reco)
def embedded_QRS_circ_reco_with_groups(params, group_params, user_params):
    for wire in item_wires:
        qml.Hadamard(wire)

    for p in user_params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)

    for p in group_params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)

    for p in params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)

    return qml.probs(wires=item_wires)



# ============================ WITH EMBEDDING GROUPS ===================================
weights_users = np.zeros((1, n_item_wires), requires_grad=False)
dev_embedded_ItemItem_reco = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_embedded_ItemItem_reco)
def embedded_QRS_circ_reco_with_groups_with_embedded(params, group_params, user_params, embedded_params):
    for wire in item_wires:
        qml.Hadamard(wire)

    for p in user_params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)

    AngleEmbedding(embedded_params, wires=item_wires, rotation='X')
    AngleEmbedding(embedded_params, wires=item_wires, rotation='Y')
    AngleEmbedding(embedded_params, wires=item_wires, rotation='Z')
    # for p in embedded_params:
    #     p = np.array([p])
    #     StronglyEntanglingLayers(p, wires=item_wires)
    # BasicEntanglerLayers(weights_users, wires=item_wires)
    # for wire in item_wires:
    #     qml.Hadamard(wire)

    for p in group_params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)
    #AngleEmbedding(group_params, wires=item_wires, rotation='Z')
    # for p in group_params:
    #     p = np.array([p])
    #     StronglyEntanglingLayers(p, wires=item_wires)

    for p in params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)

    return qml.probs(wires=item_wires)



# ============================ WITH GROUPS + HIST REMOVAL===================================
dev_embedded_ItemItem_reco = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_embedded_ItemItem_reco)
def embedded_QRS_circ_reco_hist_removal(params, group_params, user_params, embedded_params, hist_removal_params):
    for wire in item_wires:
        qml.Hadamard(wire)

    for p in user_params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)

    for p in group_params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)

    for p in params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)

    for p in hist_removal_params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)

    return qml.probs(wires=item_wires)


# noinspection PyRedundantParentheses
class embedded_QRS_model4():
    def __init__(self, R, train_steps, train_sets, layers, n_groups, user_embedding_vec):
        self.R = R
        self.train_steps = train_steps
        self.train_sets = train_sets

        self.interacted_items_matrix = self.create_interacted_items_matrix()
        self.bad_interacted_items_matrix = self.create_bad_interacted_items_matrix()
        self.uninteracted_items_matrix = self.create_uninteracted_items_matrix()
        self.expected_probs_vecs = self.create_expected_probs_vecs()

        self.user_embedding_vec = self.normalize_embdded_vecotrs(user_embedding_vec)

        self.params = self.randomize_init_params_QRS(layers)

        self.best_group_for_user = {}
        self.user_params = {}
        for user in range(defines._NUM_OF_USERS):
            self.user_params[user] = self.randomize_init_params_QRS(1)
            self.best_group_for_user[user] = 0


        #self.group_params = self.find_best_initial_group_ves(n_groups)


        self.group_params = {}
        for group in range(n_groups):
           self.group_params[group] = self.randomize_init_params_QRS(1)

        self.hist_removal_params = {}
        for user in range(defines._NUM_OF_USERS):
            self.hist_removal_params[user] = self.randomize_init_params_QRS(5)

        print(self.user_embedding_vec)
        #self.user_embedding_vec = self.conv_normalize_embdded_vecotrs_3d(self.user_embedding_vec)
        #print(self.user_embedding_vec)
        #print(self.user_embedding_vec[0])


        self.total_cost = []
        self.error_per_user = []
        for i in range(defines._NUM_OF_USERS):
            self.error_per_user.append([])

        self.recommend_phase = 0



    def __________WITHOUT_GROUPS___________(self):
        return 1

    # ----------------------------------------------------- TRAIN ------------------------------------------------------
    def train_without_groups(self):
        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.5, beta2=0.599, eps=1e-08)
        print("\n------- TRAINING EMBEDDED ITEM RECOMMENDATION SYS -------")
        for t_set in range(self.train_sets):
            print("-- TRAINING SET:", t_set, "--")

            # -------------------------- updating user params -----------------------
            params = self.params.copy()
            params.requires_grad = False
            for user in range(defines._NUM_OF_USERS):
                print("Training by user: ", user, end='\r')
                user_params = self.user_params[user]
                user_params.requires_grad = True
                for t_step in range(1):
                    params, user_params = opt_item_item.step(
                        lambda x, y: self.total_cost_embedded_QRS_without_groups(user, x, y), params, user_params)
                self.user_params[user] = user_params.copy()
            print("")

            # -------------------------- updating global params -----------------------
            params = self.params.copy()
            params.requires_grad = True
            print("Training by all users")
            for t_step in range(30):
                for user in range(defines._NUM_OF_USERS):
                    user_params = self.user_params[user]
                    user_params.requires_grad = False
                    params, user_params = opt_item_item.step(
                        lambda x, y: self.total_cost_embedded_QRS_without_groups(user, x, y), params, user_params)
            self.params = params.copy()
            print("---- DONE SET", t_set, "----\n")

        visualiser.plot_cost_arrs([self.total_cost])
        visualiser.plot_cost_arrs(self.error_per_user)


    # ----------------------------------------------------- COST -------------------------------------------------------
    def total_cost_embedded_QRS_without_groups(self, user, params, user_params):
        total_cost = 0

        # running the circuit
        probs = embedded_QRS_circ_reco_without_groups(params, user_params)
        # getting expected probs for user
        expected_probs = self.expected_probs_vecs[user]

        # calc the error for the user
        error_per_item = (expected_probs - probs) ** 2

        cost_for_user = sum(error_per_item)
        self.error_per_user[user].append(cost_for_user._value)

        total_cost += cost_for_user

        # print("total_cost:", total_cost._value)
        self.total_cost.append(total_cost._value)
        return total_cost





    def __________WITH_GROUPS___________(self):
        return 1

    # ----------------------------------------------------- TRAIN ------------------------------------------------------
    def train_with_groups(self):
        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.5, beta2=0.599, eps=1e-08)
        print("\n------- TRAINING EMBEDDED ITEM RECOMMENDATION SYS -------")
        for t_set in range(self.train_sets):
            self.total_cost.append(0)
            print("-- TRAINING SET:", t_set, "--")

            _ = self.update_best_group_for_user_dic()
            print("users groups", self.best_group_for_user)

            # -------------------------- updating user params -----------------------
            params = self.params.copy()
            params.requires_grad = False
            for user in range(defines._NUM_OF_USERS):
                print("Training by user: ", user)
                group_params = self.group_params[self.best_group_for_user[user]]
                group_params.requires_grad = False
                user_params = self.user_params[user]
                user_params.requires_grad = True
                print("user_params", user_params)
                for t_step in range(2):
                    params, group_params, user_params = opt_item_item.step(
                        lambda x, y, z: self.total_cost_embedded_QRS_with_groups(user, x, y, z), params, group_params, user_params)
                self.user_params[user] = user_params.copy()
            print("")


            # -------------------------- updating group params -----------------------
            for t_step in range(3):
                for user in range(defines._NUM_OF_USERS):
                    print("Training Group by user: ", user, end='\r')
                    group_params = self.group_params[self.best_group_for_user[user]]
                    group_params.requires_grad = True
                    user_params = self.user_params[user]
                    user_params.requires_grad = False
                    for t_step in range(3):
                        params, group_params, user_params = opt_item_item.step(
                            lambda x, y, z: self.total_cost_embedded_QRS_with_groups(user, x, y, z), params, group_params, user_params)
                    self.group_params[self.best_group_for_user[user]] = group_params.copy()
                    # if user == 15:
                    #     print("user: 15 best group", self.best_group_for_user[user],"params are:", self.group_params[self.best_group_for_user[user]])

                print("")

            # print("DEBUG:")
            # params = self.params.copy()
            # params.requires_grad = False
            # group_params = self.group_params[self.best_group_for_user[0]]
            # group_params.requires_grad = False
            # user_params = self.user_params[0]
            # user_params.requires_grad = False
            # self.total_cost_embedded_QRS_with_groups(0, params, group_params, user_params)
            # print("END DEBUG")

            # -------------------------- updating global params -----------------------
            print("Training by all users")
            for t_step in range(5):
                params = self.params.copy()
                params.requires_grad = True
                for user in range(defines._NUM_OF_USERS):
                    for t_step2 in range(5):
                        group_params = self.group_params[self.best_group_for_user[user]]
                        group_params.requires_grad = False
                        user_params = self.user_params[user]
                        user_params.requires_grad = False
                        params, group_params, user_params = opt_item_item.step(
                            lambda x, y, z: self.total_cost_embedded_QRS_with_groups(user, x, y, z), params, group_params, user_params)
                self.params = params.copy()


            # # -------------------------- updating user params -----------------------
            # params = self.params.copy()
            # params.requires_grad = False
            # for user in range(defines._NUM_OF_USERS):
            #     print("Training by user: ", user, end='\r')
            #     group_params = self.group_params[self.best_group_for_user[user]]
            #     group_params.requires_grad = False
            #     user_params = self.user_params[user]
            #     user_params.requires_grad = True
            #     for t_step in range(2):
            #         params, group_params, user_params = opt_item_item.step(
            #             lambda x, y, z: self.total_cost_embedded_QRS_with_groups(user, x, y, z), params, group_params, user_params)
            #     self.user_params[user] = user_params.copy()
            # print("")


            print("---- DONE SET", t_set, "----\n")

        visualiser.plot_cost_arrs([self.total_cost])
        visualiser.plot_cost_arrs(self.error_per_user)

    # ----------------------------------------------------- COST -------------------------------------------------------
    def total_cost_embedded_QRS_with_groups(self, user, params, group_params, user_params):

        # running the circuit
        probs = embedded_QRS_circ_reco_with_groups(params, group_params, user_params)

        # calc the error for the user
        cost_for_user = sum((self.expected_probs_vecs[user] - probs) ** 2)

        if (user ==0):
            print(probs)
        self.error_per_user[user].append(cost_for_user._value)
        self.total_cost[-1] += cost_for_user._value
        return cost_for_user






    def __________WITH_GROUPS_WITH_EMBEDDED___________(self):
        return 1

    def train(self):
        self.recommend_phase = 1  # recommend with grouping
        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.5, beta2=0.599, eps=1e-08)
        print("\n------- TRAINING EMBEDDED ITEM RECOMMENDATION SYS -------")

        params = self.params.copy()
        params.requires_grad = False

        for t_set in range(20):
            if (t_set < 10):
                _ = self.update_best_group_for_user_dic()
                print("users groups", self.best_group_for_user)

            # -------------------------- updating group params -----------------------
            for user in range(defines._NUM_OF_USERS):
                print("Training Group by user: ", user, end='\r')
                group_params = self.group_params[self.best_group_for_user[user]]
                group_params.requires_grad = True
                user_params = self.user_params[user]
                user_params.requires_grad = False
                for t_step in range(2):
                    params, group_params, user_params = opt_item_item.step(
                        lambda x, y, z: self.total_cost_embedded_QRS(user, x, y, z), params, group_params, user_params)
                self.group_params[self.best_group_for_user[user]] = group_params.copy()
            print("")


        # -------------------------- updating user params -----------------------
        for t_set in range(0):
            params = self.params.copy()
            params.requires_grad = False
            for user in range(defines._NUM_OF_USERS):
                print("Training by user: ", user, end='\r')
                group_params = self.group_params[self.best_group_for_user[user]]
                group_params.requires_grad = False
                user_params = self.user_params[user]
                user_params.requires_grad = True
                for t_step in range(1):
                    params, group_params, user_params = opt_item_item.step(
                        lambda x, y, z: self.total_cost_embedded_QRS(user, x, y, z), params, group_params, user_params)
                self.user_params[user] = user_params.copy()
            print("")

        visualiser.plot_cost_arrs([self.total_cost])
        visualiser.plot_cost_arrs(self.error_per_user)
        self.present_embedding()

    # ----------------------------------------------------- COST -------------------------------------------------------
    def total_cost_embedded_QRS(self, user, params, group_params, user_params):
        total_cost = 0
        embedded_params = self.user_embedding_vec[user]

        # running the circuit
        probs = embedded_QRS_circ_reco_with_groups_with_embedded(params, group_params, user_params, embedded_params)
        # getting expected probs for user
        expected_probs = self.expected_probs_vecs[user]

        # calc the error for the user
        error_per_item = (expected_probs - probs)**2

        cost_for_user = sum(error_per_item)
        self.error_per_user[user].append(cost_for_user._value)

        total_cost += cost_for_user

        self.total_cost.append(total_cost._value)
        return total_cost



    def get_recommendation(self, user, uninteracted_items, removed_movie):
        # get the probs vector for user

        user_params = self.user_params[user]
        group_params = self.group_params[self.best_group_for_user[user]]
        hist_removal_params = self.hist_removal_params[user]
        embedded_params = self.user_embedding_vec[user]

        if self.recommend_phase == 0:
            probs = embedded_QRS_circ_reco_with_groups(self.params, group_params, user_params)
        elif self.recommend_phase == 1:
            probs = embedded_QRS_circ_reco_with_groups_with_embedded(self.params, group_params, user_params, embedded_params)
        else:
            probs = embedded_QRS_circ_reco_hist_removal(self.params, group_params, user_params, embedded_params, hist_removal_params)
        # DEBUG
        print("recommendation for user wo hist removal:", user)
        interacted_items = self.interacted_items_matrix[user]
        bad_interacted_items = self.bad_interacted_items_matrix[user]
        visualiser.print_colored_matrix(probs, [bad_interacted_items, interacted_items, np.array([removed_movie])],
                                        is_vec=1,
                                        all_positive=1, digits_after_point=2)
        return probs

    def find_best_embedding_group_for_user(self, user):
        min_error = 2
        best_group = 0
        user_params = self.user_params[user]
        expected_probs = self.expected_probs_vecs[user]
        embedded_params = self.user_embedding_vec[user]
        for group, group_params in self.group_params.items():
            probs = embedded_QRS_circ_reco_with_groups(self.params, group_params, user_params)
            total_error = sum((expected_probs - probs) ** 2)
            if total_error < min_error:
                min_error = total_error
                best_group = group
        return best_group

    # for evey user - find the best group that fit him
    # return: max diff between groups sizes
    def update_best_group_for_user_dic(self):
        num_of_user_per_groups = np.zeros(len(self.group_params))
        for user in range(defines._NUM_OF_USERS):
            new_group_for_user = self.find_best_embedding_group_for_user(user)
            num_of_user_per_groups[new_group_for_user] += 1
            if new_group_for_user != self.best_group_for_user[user]:
                print("user", user, "changed group", self.best_group_for_user[user], "->", new_group_for_user)
                self.best_group_for_user[user] = new_group_for_user
        return max(num_of_user_per_groups) - min(num_of_user_per_groups)


    # def get_QRS_reco_matrix(self):
    #     QRS_reco_matrix = []
    #     for user in range(defines._NUM_OF_USERS):
    #         user_params = self.user_params[user]
    #         group_params = self.group_params[self.best_group_for_user[user]]
    #         probs = embedded_QRS_circ_reco(self.params, group_params, user_params)
    #         QRS_reco_matrix.append(probs)
    #     return QRS_reco_matrix

    # plotting the embedded quantum states:
    def present_embedding(self):
        embedded_state = []
        for user in range(defines._NUM_OF_USERS):
            user_params = self.user_params[user]
            probs = embedded_QRS_circ_embedding(user_params)
            embedded_state.append(probs)

        for group, group_params in self.group_params.items():
            probs = embedded_QRS_circ_embedding(group_params)
            embedded_state.append(probs)

        visualiser.plot_embedded_vecs(embedded_state)

    def get_embedding_vecs(self):
        return self.user_params

    def get_QRS_opt_params(self):
        return self.params

    def create_interacted_items_matrix(self):
        expected_mat = []
        for user in range(defines._NUM_OF_USERS):
            items = np.where(self.R[user] == 1)[0]
            expected_mat.append(items)
        return expected_mat

    def create_bad_interacted_items_matrix(self):
        expected_mat = []
        for user in range(defines._NUM_OF_USERS):
            items = np.where(self.R[user] == defines._BAD_SAMPLED_INTER)[0]
            expected_mat.append(items)
        return expected_mat

    def create_uninteracted_items_matrix(self):
        expected_mat = []
        for user in range(defines._NUM_OF_USERS):
            interacted_items = self.interacted_items_matrix[user]
            bad_interacted_items = self.bad_interacted_items_matrix[user]
            uninteracted_items = [i for i in range(defines._NUM_OF_ITEMS) if
                                  i not in interacted_items and i not in bad_interacted_items]
            expected_mat.append(uninteracted_items)
        return expected_mat


    def create_expected_probs_vecs(self):
        expected_probs_vecs = []
        for user in range(defines._NUM_OF_USERS):

            # getting the indecies where user have positive interaction
            interacted_items = self.interacted_items_matrix[user]

            # getting the indecies where user have negetive interaction
            bad_interacted_items = self.bad_interacted_items_matrix[user]

            # building the expected prop array
            # for interacted items - the expected val is _MAX_HIST_INTER_WEIGHT/(num of interacted items)
            # for un-interacted items - the expected val is (1-_MAX_HIST_INTER_WEIGHT)/ num of un-interacted items
            # for bad-interacted items - the expected val is 0
            expected_probs = np.ones(defines._NUM_OF_ITEMS, requires_grad=False) * (
                        1 - defines._MAX_HIST_INTER_WEIGHT) / (
                                     defines._NUM_OF_ITEMS - len(interacted_items) - len(bad_interacted_items))
            if (len(interacted_items) > 0):
                expected_probs[interacted_items] = defines._MAX_HIST_INTER_WEIGHT / len(interacted_items)
            expected_probs[bad_interacted_items] = 0

            expected_probs_vecs.append(expected_probs)

        return expected_probs_vecs



    # input: list of vectors
    # output: list of vectors which all positive, and the sum of each vector is smaller than pi
    def normalize_embdded_vecotrs(self, vecs):
        columns_mins = vecs.min(axis=0)
        vecs -= columns_mins     # min in evey columns is 0

        global_max = vecs.max()+0.0001
        vecs /= global_max  # max in all data is 1

        vecs *= (2*math.pi/defines._EMBEDDING_SIZE)       # sum of each row is up to pi
        return vecs

    def conv_normalize_embdded_vecotrs_3d(self, vecs):
        conv_norm_vecs = {}
        for i, v in enumerate(vecs):
            v3d = np.array([v, v, v])
            v3d_T = v3d.T
            conv_norm_vecs[i] = np.array([v3d_T], requires_grad=False)
        return conv_norm_vecs


    # creating nof_groups initial vectors - which suppose to split the users to seperated groups
    def find_best_initial_group_ves(self, nof_groups):
        closest_group_vec_to_each_vec = np.zeros(nof_groups, requires_grad=False)
        group_vecs = np.zeros((nof_groups, len(self.user_embedding_vec[0])), requires_grad=False)
        # finding the most nof_groups farest vectors
        farest_vecs = self.find_farest_vecs_from_list(nof_groups, self.user_embedding_vec, 100)
        for user in range(defines._NUM_OF_USERS):
            v = self.user_embedding_vec[user]
            # for everry user finding the closest vecs within the farest which were chosen
            closest_vec_ind = self.find_closest_vec_to_vec(v, farest_vecs)
            print("user:", user, "closest vec:", closest_vec_ind)
            group_vecs[closest_vec_ind] -= v
            closest_group_vec_to_each_vec[closest_vec_ind] += 1

        for i in range(nof_groups):
            group_vecs[i] /= closest_group_vec_to_each_vec[i]

        group_vecs = self.conv_normalize_embdded_vecotrs_3d(group_vecs)

        return group_vecs



    # from a list of vecs (vecs_list) - choosing nof_vecs_to_sample with the largest distance
    def find_farest_vecs_from_list(self, nof_vecs_to_sample, vecs_list, iterations):
        best_total_dist = 0
        best_sampled_vecs = []
        for iter in range(iterations):
            total_dist = 0
            sampled_vecs = random.sample(list(vecs_list), nof_vecs_to_sample)
            for i in range(nof_vecs_to_sample):
                for j in range(i+1, nof_vecs_to_sample):
                    total_dist += math.sqrt(sum((sampled_vecs[i] - sampled_vecs[j])**2))
            if (total_dist > best_total_dist):
                best_total_dist = total_dist
                best_sampled_vecs = sampled_vecs.copy()

        print("best total dist:", best_total_dist, "vecs:", best_sampled_vecs)
        dic_best_sampled_vecs = {i:np.array(v) for i,v in enumerate(best_sampled_vecs)}
        return dic_best_sampled_vecs


    # choosing from list of vecs (vecs) the closest vector index to input vec (vec)
    def find_closest_vec_to_vec(self, vec, vecs):
        closest_vec_ind = 0
        closest_dist = 100000000
        for i,v in enumerate(vecs):
            dist = math.sqrt(sum((vec - v)**2))
            if (dist < closest_dist):
                closest_vec_ind = i
                closest_dist = dist
        return closest_vec_ind


    def __________HIST_REMOVAL___________(self):
        return 1


    def get_QRS_reco_matrix(self):
        QRS_reco_matrix = []
        for user in range(defines._NUM_OF_USERS):
            user_params = self.user_params[user]
            group_params = self.group_params[self.best_group_for_user[user]]
            probs = embedded_QRS_circ_reco_with_groups(self.params, group_params, user_params)
            QRS_reco_matrix.append(probs)
        return QRS_reco_matrix


    # calculating the expected probs for evey user
    # the expected probs for user - is the ralted probs for item - but after zeroing the interacted items
    def create_expected_probs_vecs_for_hist_removal(self):
        QRS_reco_matrix = self.get_QRS_reco_matrix()
        expected_probs_vecs = []
        for user in range(defines._NUM_OF_USERS):
            # getting the indecies where user have positive interaction
            interacted_items =self.interacted_items_matrix[user]
            bad_interacted_items = self.bad_interacted_items_matrix[user]

            # gertting the recommendations from QRS circ
            expected_probs = QRS_reco_matrix[user]

            # calc the expected probs for user
            expected_probs[interacted_items] = 0
            expected_probs[bad_interacted_items] = 0
            expected_probs = expected_probs/sum(expected_probs)

            expected_probs_vecs.append(expected_probs)

        return expected_probs_vecs

    def train_hist_removal(self):
        self.error_per_user = []
        for i in range(defines._NUM_OF_USERS):
            self.error_per_user.append([])

        self.expected_probs_vecs_for_hist_removal = self.create_expected_probs_vecs_for_hist_removal()

        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.5, beta2=0.599, eps=1e-08)
        for user in range(defines._NUM_OF_USERS):
            print("TRAINING QRS HIST REMOVAL user:", user, end='\r')

            hist_removal_params = self.hist_removal_params[user].copy()
            hist_removal_params.requires_grad = True
            for i in range(self.train_steps):
                hist_removal_params = opt_item_item.step(lambda v: self.total_cost_embedded_QRS_hist_removal(user, v),
                                                         hist_removal_params)
            self.hist_removal_params[user] = hist_removal_params.copy()

        print("")
        self.recommend_phase = 2
        visualiser.plot_cost_arrs(self.error_per_user)


    def total_cost_embedded_QRS_hist_removal(self, user, hist_params):
        # running the circuit
        user_params = self.user_params[user]
        group_params = self.group_params[self.best_group_for_user[user]]
        embedded_params = self.user_embedding_vec[user]
        probs = embedded_QRS_circ_reco_hist_removal(self.params, group_params, user_params, embedded_params, hist_params)

        expected_probs = self.expected_probs_vecs_for_hist_removal[user]
        error_per_item = (expected_probs - probs)**2

        interacted_items = self.interacted_items_matrix[user]
        bad_interacted_items = self.bad_interacted_items_matrix[user]
        uninteracted_items = self.uninteracted_items_matrix[user]


        for i in interacted_items:
            error_per_item[i]._value = error_per_item[i]._value * 2
        for i in bad_interacted_items:
            error_per_item[i]._value = error_per_item[i]._value * 10
        for i in uninteracted_items:
            error_per_item[i]._value = error_per_item[i]._value * (expected_probs[i]*100)

        cost_for_user = sum(error_per_item)
        self.error_per_user[user].append(cost_for_user._value)
        return cost_for_user




    def randomize_init_params_QRS(self, layers):
        shape = StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_item_wires)
        return np.random.random(size=shape, requires_grad=False)
