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


dev_QRS_user_items_circ = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_QRS_user_items_circ)
def QRS_user_items_circ(params):
    for wire in item_wires:
        qml.Hadamard(wire)

    for p in params:
        p = np.array(p)
        StronglyEntanglingLayers(p, wires=item_wires)

    return qml.probs(wires=item_wires)


class QRS_user_items():
    def __init__(self, R):
        self.R = R

        self.interacted_items_matrix = self.create_interacted_items_matrix()
        self.bad_interacted_items_matrix = self.create_bad_interacted_items_matrix()
        self.uninteracted_items_matrix = self.create_uninteracted_items_matrix()
        self.expected_probs_vecs = self.create_expected_probs_vecs()
        self.users_interacted_with_item_matrix = self.build_users_interacted_with_item_matrix()


        self.user_params = {}
        for user in range(defines._NUM_OF_USERS):
            self.user_params[user] = self.randomize_init_params_QRS(1)

        self.item_params = {}
        for item in range(defines._NUM_OF_ITEMS):
            self.item_params[item] = self.randomize_init_params_QRS(1)

        self.hist_removal_params = {}
        for user in range(defines._NUM_OF_USERS):
            self.hist_removal_params[user] = self.randomize_init_params_QRS(1)

        self.total_cost = []
        self.error_per_user = []
        for i in range(defines._NUM_OF_USERS):
            self.error_per_user.append([])

        self.hist_removal_en = 0

    # ----------------------------------------------------- TRAIN ------------------------------------------------------
    def train(self):
        opt_item_item = qml.AdamOptimizer(stepsize=0.005, beta1=0.9, beta2=0.99, eps=1e-08)

        print("\n------- TRAINING RECOMMENDATION SYS -------")
        for set in range(7):
            print("--- Training set", set, " ---")
            self.total_cost.append(0)

            # ---------------------------- TRAINING BY USER ----------------------------
            if (set%2==0):
                self.set_all_require_grad_false()
                for user in range(defines._NUM_OF_USERS):
                    print("Training by user: ", user, end='\r')
                    params, grad_param_pos = self.get_params(user, -1)
                    for t_step in range(2):
                        params = opt_item_item.step(self.total_cost_QRS_user_items, *params, user=user)
                    self.update_params(user, params)
                print("")


            # ---------------------------- TRAINING BY ITEM ----------------------------
            self.set_all_require_grad_false()
            for item in range(defines._NUM_OF_ITEMS):
                print("Training item", item, end='\r')
                for user in self.users_interacted_with_item_matrix[item]:
                    params, grad_param_pos = self.get_params(user, item)
                    for t_step in range(20):
                        params = opt_item_item.step(self.total_cost_QRS_user_items, *params, user=user, item=item)
                    self.update_params(user, params)
            print(f"total cost: {self.total_cost[-1]:.3f}\n")


        visualiser.plot_cost_arrs([self.total_cost])
        visualiser.plot_cost_arrs(self.error_per_user)


    # ----------------------------------------------------- COST -------------------------------------------------------
    def total_cost_QRS_user_items(self, *params, **kwargs):

        probs = QRS_user_items_circ(params)

        if self.hist_removal_en == 0:
            expected_probs = self.expected_probs_vecs[kwargs['user']]
            for item in self.uninteracted_items_matrix[kwargs['user']]:
                expected_probs[item] = probs[item]._value
            # calc the error for the user
            # cost_for_user = self.cross_entropy_loss(probs, kwargs['user'])
            cost_for_user = sum((expected_probs - probs) ** 2)

        if self.hist_removal_en == 1:
            expected_probs = self.expected_probs_vecs_for_hist_removal[kwargs['user']]
            # calc the error for the user
            cost_for_user = sum((expected_probs - probs) ** 2)



        self.error_per_user[kwargs['user']].append(cost_for_user._value)

        self.total_cost[-1] += cost_for_user._value
        return cost_for_user


    def get_params(self, user, item_optimized):
        grad_param_pos = 0
        params = []
        interacted_items_list = self.interacted_items_matrix[user]

        if (self.hist_removal_en):
            t = self.user_params[user].copy()
            t.requires_grad = False
            params.append(t)
            for item in interacted_items_list:
                t = self.item_params[item].copy()
                t.requires_grad = False
                params.append(t)
            grad_param_pos = len(params)
            t = self.hist_removal_params[user].copy()
            t.requires_grad = True
            params.append(t)
            return params, grad_param_pos



        # training by user:
        if (item_optimized == -1):
            t = self.user_params[user].copy()
            t.requires_grad = True
            params.append(t)
            for item in interacted_items_list:
                t = self.item_params[item].copy()
                t.requires_grad = False
                params.append(t)

        # training by item:
        else:
            t = self.user_params[user].copy()
            t.requires_grad = False
            params.append(t)
            for i, item in enumerate(interacted_items_list):
                t = self.item_params[item].copy()
                if (item == item_optimized):
                    grad_param_pos = len(params)
                    t.requires_grad = True
                else:
                    t.requires_grad = False
                params.append(t)

        return params, grad_param_pos


    def update_params(self, user, params):
        self.user_params[user] = params[0].copy()
        for i,item in enumerate(self.interacted_items_matrix[user]):
            self.item_params[item] = params[i+1].copy()
        if (self.hist_removal_en):
            self.hist_removal_params[user] = params[-1].copy()



    def get_recommendation(self, user, uninteracted_items, removed_movie):

        params, grad_param_pos = self.get_params(user, -1)

        probs = QRS_user_items_circ(params)

        # DEBUG
        print("recommendation for user:", user)
        interacted_items = self.interacted_items_matrix[user]
        bad_interacted_items = self.bad_interacted_items_matrix[user]
        visualiser.print_colored_matrix(probs, [bad_interacted_items, interacted_items, np.array([removed_movie])],
                                        is_vec=1,
                                        all_positive=1, digits_after_point=2)
        return probs




    def __________HIST_REMOVAL___________(self):
        return 1



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
        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.5, beta2=0.599, eps=1e-08)
        self.expected_probs_vecs_for_hist_removal = self.create_expected_probs_vecs_for_hist_removal()

        self.hist_removal_en = 1

        self.set_all_require_grad_false()
        for user in range(defines._NUM_OF_USERS):
            print("Hist Removal Training by user: ", user, end='\r')
            params, grad_param_pos = self.get_params(user, -1)
            for i in range(50):
                params = opt_item_item.step(self.total_cost_QRS_user_items, *params, user=user)
            self.update_params(user, params)
        print("")

        print("")
        visualiser.plot_cost_arrs(self.error_per_user)



    def _________AUX_FUNCTIONS___________(self):
        return 1


    def randomize_init_params_QRS(self, layers):
        shape = StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_item_wires)
        return np.random.random(size=shape, requires_grad=False)


    def get_QRS_reco_matrix(self):
        QRS_reco_matrix = []
        for user in range(defines._NUM_OF_USERS):
            params, grad_param_pos = self.get_params(user, -1)
            probs = QRS_user_items_circ(params)  # no hist removal
            QRS_reco_matrix.append(probs)
        return QRS_reco_matrix



    # item_user_interaction_mat[i] = list(all users interacted with item i)
    def build_users_interacted_with_item_matrix(self):
        item_user_interaction_mat = []
        for item in range(defines._NUM_OF_ITEMS):
            item_user_interaction_mat.append([])
            for user in range(defines._NUM_OF_USERS):
                if item in self.interacted_items_matrix[user]:
                    item_user_interaction_mat[item].append(user)
        return item_user_interaction_mat


    def set_all_require_grad_false(self):
        for item in self.item_params:
            self.item_params[item].require_grad = False
        for user in self.user_params:
            self.user_params[user].require_grad = False



    def get_QRS_opt_params(self):
        return self.item_params

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

    def cross_entropy_loss(self, probs, user):
        loss = 0
        for item in self.interacted_items_matrix[user]:
            loss -= np.log(probs[item]/0.33)
        for item in self.uninteracted_items_matrix[user]:
            loss -= np.log(1-probs[item])
        return loss


