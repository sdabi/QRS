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


dev_basic_QRS_circ = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_basic_QRS_circ)
def basic_QRS_circ(params):
    for wire in item_wires:
        qml.Hadamard(wire)
    for p in params:
        p = np.array([p])
        StronglyEntanglingLayers(p, wires=item_wires)

    return qml.probs(wires=item_wires)




class basic_QRS():
    def __init__(self, R):
        self.R = R

        self.interacted_items_matrix           = self.create_interacted_items_matrix()
        self.bad_interacted_items_matrix       = self.create_bad_interacted_items_matrix()
        self.uninteracted_items_matrix         = self.create_uninteracted_items_matrix()
        self.expected_probs_vecs               = self.create_expected_probs_vecs()
        self.users_interacted_with_item_matrix = self.build_users_interacted_with_item_matrix()
        self.cost_ampl_mask_matrix             = self.create_cost_amplification_mask_matrix()

        self.total_cost = []
        self.error_per_user = []
        for i in range(defines._NUM_OF_USERS):
            self.error_per_user.append([])

        self.hist_removal_en = 0




    def randomize_init_params_QRS(self, layers):
        shape = StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_item_wires)
        return np.random.random(size=shape, requires_grad=False)


    # calculating the expected probs for evey user
    # the expected probs for user - is the ralted probs for item - but after zeroing the interacted items
    def create_expected_probs_vecs_for_hist_removal(self, QRS_reco_matrix):
        expected_probs_vecs = []
        for user in range(defines._NUM_OF_USERS):

            # gertting the recommendations from QRS circ
            expected_probs = QRS_reco_matrix[user]

            # calc the expected probs for user
            expected_probs[self.interacted_items_matrix[user]] = 0
            expected_probs[self.bad_interacted_items_matrix[user]] = 0
            expected_probs = expected_probs/sum(expected_probs)

            expected_probs_vecs.append(expected_probs)

        return expected_probs_vecs


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

    # item_user_interaction_mat[i] = list(all users interacted with item i)
    def build_users_interacted_with_item_matrix(self):
        item_user_interaction_mat = []
        for item in range(defines._NUM_OF_ITEMS):
            item_user_interaction_mat.append([])
            for user in range(defines._NUM_OF_USERS):
                if item in self.interacted_items_matrix[user]:
                    item_user_interaction_mat[item].append(user)
        return item_user_interaction_mat


    # calculating the cost on the basic_QRS_cric
    # getting as inputs:  *params = pointer to list of parameters and user=user, item=item expected_probs=expected_probs
    # returns optimized params according to requires_grad field
    # example to a call:
    # params = opt_item_item.step(self.total_cost_basic_QRS_user_items, *params, user=user, item=item, expected_probs=expected_probs)
    def total_cost_basic_QRS_user_items(self, *params, **kwargs):

        probs = basic_QRS_circ(params)
        cost_for_user = sum(((kwargs['expected_probs'] - probs) ** 2)*(kwargs['cost_ampl_mask']))

        self.error_per_user[kwargs['user']].append(cost_for_user._value)
        self.total_cost[-1] += cost_for_user._value

        # if (kwargs['user'] == 0):
        #     print("DEBUG COST user", kwargs['user'])
        #     print(kwargs['expected_probs'])
        #     print(probs._value)
        #     print(kwargs['cost_ampl_mask'])
        #     err = ((kwargs['expected_probs'] - probs) ** 2)*(kwargs['cost_ampl_mask'])
        #     print(err._value)
        return cost_for_user

    def create_cost_amplification_mask_matrix(self):
        cost_ampl_mask_matrix = []
        for user in range(defines._NUM_OF_USERS):
            cost_ampl_mask = np.ones((defines._NUM_OF_ITEMS)) * 1
            cost_ampl_mask[self.uninteracted_items_matrix[user]] = 0
            cost_ampl_mask[self.bad_interacted_items_matrix[user]] = 0
            cost_ampl_mask_matrix.append(cost_ampl_mask)
        return cost_ampl_mask_matrix

    # need to be overriden in every class
    def get_recommendation(self, user, uninteracted_items, removed_movie, debug=1):
        return np.zeros(defines._NUM_OF_ITEMS)


    def get_QRS_reco_matrix(self):
        QRS_reco_matrix = []
        for user in range(defines._NUM_OF_USERS):
            probs = self.get_recommendation(user, 0, 0, 0)
            QRS_reco_matrix.append(probs)
        return QRS_reco_matrix



