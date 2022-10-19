import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.templates.layers import BasicEntanglerLayers
from pennylane import numpy as np
import time

import math
import defines
import visualiser

n_item_wires = (int(math.log(defines._NUM_OF_ITEMS,2)))
n_user_wires = (int(math.log(defines._NUM_OF_USERS,2)))
n_wires = n_user_wires + n_item_wires

item_wires = list(range(n_item_wires))
user_wires = list(range(n_item_wires, n_wires))
wires_list = item_wires + user_wires

weights_users = np.zeros((1, n_item_wires), requires_grad=False)


# wrap device in qml.qnode
dev_embedded_ItemItem_reco = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_embedded_ItemItem_reco)
def QRS_hist_removal_circ(embedded_params, QRS_opt_params, params):
    # QRS
    for p in QRS_opt_params:
        p = np.array([p])
        for wire in item_wires:
            qml.Hadamard(wire)
        AngleEmbedding(embedded_params, wires=item_wires, rotation='Z')
        BasicEntanglerLayers(weights_users, wires=item_wires) # entanglement
        for wire in item_wires:
            qml.Hadamard(wire)
        StronglyEntanglingLayers(p, wires=item_wires)

    # Hist Removal
    StronglyEntanglingLayers(params, wires=wires_list)
    return qml.probs(wires=item_wires)


def randomize_init_params():
    shape = StronglyEntanglingLayers.shape(n_layers=defines._NUM_OF_LAYERS, n_wires=n_wires)
    return np.random.random(size=shape, requires_grad=True)


class QRS_hist_removal():
    def __init__(self, R, QRS_reco_matrix, user_embedded_vecs, QRS_opt_params, train_steps):
        self.R = R
        self.QRS_reco_matrix = QRS_reco_matrix
        self.params = randomize_init_params()
        self.train_steps = train_steps
        self.user_embedded_vecs = user_embedded_vecs
        self.QRS_opt_params = QRS_opt_params

        self.interacted_items_matrix = self.create_interacted_items_matrix()
        self.bad_interacted_items_matrix = self.create_bad_interacted_items_matrix()
        self.uninteracted_items_matrix = self.create_uninteracted_items_matrix()
        self.expected_probs_vecs = self.create_expected_probs_vecs()

        self.total_cost = []
        self.error_per_user = []
        for i in range(defines._NUM_OF_USERS):
            self.error_per_user.append([])

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


    # calculating the expected probs for evey user
    # the expected probs for user - is the ralted probs for item - but after zeroing the interacted items
    def create_expected_probs_vecs(self):
        expected_probs_vecs = []
        for user in range(defines._NUM_OF_USERS):
            # getting the indecies where user have positive interaction
            interacted_items =self.interacted_items_matrix[user]
            bad_interacted_items = self.bad_interacted_items_matrix[user]

            # gertting the recommendations from QRS circ
            expected_probs = self.QRS_reco_matrix[user]

            if user == 0:
                print("expected before HR: ", expected_probs)

            # calc the expected probs for user
            expected_probs[interacted_items] = 0
            expected_probs[bad_interacted_items] = 0
            expected_probs = expected_probs/sum(expected_probs)

            if user == 0:
                print("expected after HR: ", expected_probs)

            expected_probs_vecs.append(expected_probs)

        return expected_probs_vecs

    def train(self):
        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.5, beta2=0.599, eps=1e-08)
        start_time_train = time.time()
        print("\n------- TRAINING EMBEDDED ITEM RECOMMENDATION SYS -------")

        for i in range(self.train_steps):
            if i % 10 == 0:
                print("training step:", i)
            self.params = opt_item_item.step(lambda v: self.total_cost_embedded_QRS(v), self.params)

        print("--- embedding train took %s seconds ---" % math.ceil(time.time() - start_time_train))
        visualiser.plot_cost_arrs([self.total_cost])
        visualiser.plot_cost_arrs(self.error_per_user)


    def total_cost_embedded_QRS(self, params):
        total_cost = 0
        for user in list(range(defines._NUM_OF_USERS)):
            embedded_vec_for_user = self.user_embedded_vecs[user]

            # running the circuit
            probs = QRS_hist_removal_circ(embedded_vec_for_user, self.QRS_opt_params, params)

            expected_probs = self.expected_probs_vecs[user]
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
            total_cost += cost_for_user

            # DEBUG:
            if user == 0:
                print("user:", user)
                visualiser.print_colored_matrix(expected_probs, [bad_interacted_items, interacted_items], is_vec=1,
                                                all_positive=1, digits_after_point=2)
                visualiser.print_colored_matrix(probs._value, [bad_interacted_items, interacted_items], is_vec=1,
                                                all_positive=1, digits_after_point=2)
                visualiser.print_colored_matrix(error_per_item._value, [bad_interacted_items, interacted_items], is_vec=1,
                                                all_positive=1, digits_after_point=2)

                print("cost_for_user", cost_for_user._value, "\n")

        print("total_cost:", total_cost._value)
        print("------------------------------------","\n")
        self.total_cost.append(total_cost._value)
        return total_cost


    def get_recommendation(self, user, uninteracted_items, removed_movie):

        embedded_vec_for_user = self.user_embedded_vecs[user]

        # get the probs vector for user
        probs = QRS_hist_removal_circ(embedded_vec_for_user, self.QRS_opt_params, self.params)

        # DEBUG
        print("recommendation for user after hist removal:", user)
        interacted_items = self.interacted_items_matrix[user]
        bad_interacted_items = self.bad_interacted_items_matrix[user]
        visualiser.print_colored_matrix(probs, [bad_interacted_items, interacted_items, np.array([removed_movie])], is_vec=1,
                                        all_positive=1, digits_after_point=2)

        return probs



