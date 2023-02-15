import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane import numpy as np
import time
import itertools

import math
import defines
import visualiser

n_item_wires = (int(math.log(defines._NUM_OF_ITEMS,2)))
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




# wrap device in qml.qnode
dev_embedded_ItemItem_reco = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_embedded_ItemItem_reco)
def embedded_QRS_circ_reco(params, group_params, user_params):

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


def randomize_init_params_QRS2(layers):
    shape = StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_item_wires)
    return np.random.random(size=shape, requires_grad=False)


class embedded_QRS_model3():
    def __init__(self, R, train_steps, train_sets, layers, n_groups):
        self.R = R
        self.train_steps = train_steps
        self.train_sets = train_sets

        self.interacted_items_matrix = self.create_interacted_items_matrix()
        self.bad_interacted_items_matrix = self.create_bad_interacted_items_matrix()
        self.uninteracted_items_matrix = self.create_uninteracted_items_matrix()
        self.expected_probs_vecs = self.create_expected_probs_vecs()

        self.params = randomize_init_params_QRS2(layers)
        self.user_params = {}
        self.best_group_for_user = {}
        for user in range(defines._NUM_OF_USERS):
            self.user_params[user] = randomize_init_params_QRS2(1)
            self.best_group_for_user[user] = 0

        self.group_params = {}
        for group in range(n_groups):
            self.group_params[group] = randomize_init_params_QRS2(1)


        self.total_cost = []
        self.error_per_user = []
        for i in range(defines._NUM_OF_USERS):
            self.error_per_user.append([])


    # ----------------------------------------------------- TRAIN ------------------------------------------------------
    def train(self):
        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.5, beta2=0.599, eps=1e-08)
        start_time_train = time.time()
        print("\n------- TRAINING EMBEDDED ITEM RECOMMENDATION SYS -------")
        for t_set in range(self.train_sets):
            print("-- TRAINING SET:", t_set, "--")

            self.update_best_group_for_user_dic()

            # -------------------------- updating user params -----------------------
            params = self.params.copy()
            params.requires_grad = False
            for user in range(defines._NUM_OF_USERS):
                print("Training by user: ", user, end='\r')
                group_params = self.group_params[self.best_group_for_user[user]]
                group_params.requires_grad = False
                user_params = self.user_params[user]
                user_params.requires_grad = True
                for t_step in range(10):
                    params, group_params, user_params = opt_item_item.step(
                        lambda x, y, z: self.total_cost_embedded_QRS(user, x, y, z), params, group_params, user_params)
                self.user_params[user] = user_params.copy()
            print("")

            self.update_best_group_for_user_dic()

            # -------------------------- updating group params -----------------------
            params = self.params.copy()
            params.requires_grad = False
            for user in range(defines._NUM_OF_USERS):
                print("Training Group by user: ", user, end='\r')
                group_params = self.group_params[self.best_group_for_user[user]]
                group_params.requires_grad = True
                user_params = self.user_params[user]
                user_params.requires_grad = False
                for t_step in range(7):
                    params, group_params, user_params = opt_item_item.step(
                        lambda x, y, z: self.total_cost_embedded_QRS(user, x, y, z), params, group_params, user_params)
                self.group_params[self.best_group_for_user[user]] = group_params.copy()
            print("")

            self.update_best_group_for_user_dic()
            # -------------------------- updating global params -----------------------
            params = self.params.copy()
            params.requires_grad = True
            print("Training by all users")
            for t_step in range(5):
                for user in range(defines._NUM_OF_USERS):
                    group_params = self.group_params[self.best_group_for_user[user]]
                    group_params.requires_grad = False
                    user_params = self.user_params[user]
                    user_params.requires_grad = False
                    params, group_params, user_params = opt_item_item.step(
                        lambda x, y, z: self.total_cost_embedded_QRS(user, x, y, z), params, group_params, user_params)
            self.params = params.copy()

            print("---- DONE SET", t_set, "----\n")

        print("--- embedding train took %s seconds ---" % math.ceil(time.time() - start_time_train))
        #self.update_best_group_for_user_dic()
        visualiser.plot_cost_arrs([self.total_cost])
        visualiser.plot_cost_arrs(self.error_per_user)

        self.present_embedding()

    # ----------------------------------------------------- COST -------------------------------------------------------
    def total_cost_embedded_QRS(self, user, params, group_params, user_params):
        total_cost = 0

        # running the circuit
        probs = embedded_QRS_circ_reco(params, group_params, user_params)
        # getting expected probs for user
        expected_probs = self.expected_probs_vecs[user]

        # calc the error for the user
        error_per_item = (expected_probs - probs)**2

        cost_for_user = sum(error_per_item)
        self.error_per_user[user].append(cost_for_user._value)

        total_cost += cost_for_user

        #print("total_cost:", total_cost._value)
        self.total_cost.append(total_cost._value)
        return total_cost



    def get_recommendation(self, user, uninteracted_items, removed_movie):
        # get the probs vector for user

        user_params = self.user_params[user]
        group_params = self.group_params[self.best_group_for_user[user]]
        probs = embedded_QRS_circ_reco(self.params, group_params, user_params)
        # DEBUG
        print("recommendation for user wo hist removal:", user)
        interacted_items = self.interacted_items_matrix[user]
        bad_interacted_items = self.bad_interacted_items_matrix[user]
        visualiser.print_colored_matrix(probs, [bad_interacted_items, interacted_items, np.array([removed_movie])], is_vec=1,
                                        all_positive=1, digits_after_point=2)
        return probs


    def find_best_embedding_group_for_user(self, user):
        min_error = 2
        best_group = 0
        user_params = self.user_params[user]
        expected_probs = self.expected_probs_vecs[user]
        for group, group_params in self.group_params.items():
            probs = embedded_QRS_circ_reco(self.params, group_params, user_params)
            error_per_item = (expected_probs - probs) ** 2
            total_error = sum(error_per_item)
            if total_error < min_error:
                min_error = total_error
                best_group = group
        return best_group


    def update_best_group_for_user_dic(self):
        for user in range(defines._NUM_OF_USERS):
            new_group_for_user = self.find_best_embedding_group_for_user(user)
            if new_group_for_user != self.best_group_for_user[user]:
                print("user", user, "changed group", self.best_group_for_user[user], "->", new_group_for_user)
                self.best_group_for_user[user] = new_group_for_user
            #print("user:", user, "belongs to group", self.best_group_for_user[user])



    def get_QRS_reco_matrix(self):
        QRS_reco_matrix = []
        for user in range(defines._NUM_OF_USERS):
            user_params = self.user_params[user]
            group_params = self.group_params[self.best_group_for_user[user]]
            probs = embedded_QRS_circ_reco(self.params, group_params, user_params)
            QRS_reco_matrix.append(probs)
        return QRS_reco_matrix


    # plotting the embedded quantum states:
    def present_embedding(self):
        embedded_state = []
        for user in range(defines._NUM_OF_USERS):
            user_params = self.user_params[user]
            probs = embedded_QRS_circ_embedding(user_params)
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
            interacted_items     = self.interacted_items_matrix[user]
            bad_interacted_items = self.bad_interacted_items_matrix[user]
            uninteracted_items   = [i for i in range(defines._NUM_OF_ITEMS) if
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
            expected_probs = np.ones(defines._NUM_OF_ITEMS, requires_grad=False) * (1 - defines._MAX_HIST_INTER_WEIGHT) / (
                        defines._NUM_OF_ITEMS - len(interacted_items) - len(bad_interacted_items))
            if (len(interacted_items) > 0):
                expected_probs[interacted_items] = defines._MAX_HIST_INTER_WEIGHT / len(interacted_items)
            expected_probs[bad_interacted_items] = 0

            expected_probs_vecs.append(expected_probs)

        return expected_probs_vecs



