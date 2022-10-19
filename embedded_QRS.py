import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.templates.layers import BasicEntanglerLayers
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

# weights =  np.random.random((1, n_wires))
weights_users = np.zeros((1, n_item_wires), requires_grad=False)
weights_all = np.zeros((1, n_wires), requires_grad=False)



# this cricuit is used to plot the embeding quantum state only
dev_embedded_ItemItem_embedding = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_embedded_ItemItem_embedding)
def embedded_QRS_circ_embedding(embedded_params):
    for wire in item_wires:
        qml.Hadamard(wire)
    AngleEmbedding(embedded_params, wires=item_wires, rotation='Z')
    BasicEntanglerLayers(weights_users, wires=item_wires)
    for wire in item_wires:
        qml.Hadamard(wire)
    return qml.probs(wires=item_wires)




# wrap device in qml.qnode
dev_embedded_ItemItem_reco = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_embedded_ItemItem_reco)
# def circuit(params,state=None):
def embedded_QRS_circ_reco(embedded_params, params):
    for p in params:
        p = np.array([p])
        for wire in item_wires:
            qml.Hadamard(wire)
        AngleEmbedding(embedded_params, wires=item_wires, rotation='Z')
        BasicEntanglerLayers(weights_users, wires=item_wires)
        for wire in item_wires:
            qml.Hadamard(wire)

        # BasicEntanglerLayers(weights_all, wires=wires_list)
        StronglyEntanglingLayers(p, wires=item_wires)

    return qml.probs(wires=item_wires)


def randomize_init_params(num_of_layers_l):
    init_params_list = []
    for num_of_layers in num_of_layers_l:
        shape = StronglyEntanglingLayers.shape(n_layers=num_of_layers, n_wires=n_item_wires)
        init_params_list.append(np.random.random(size=shape, requires_grad=True))
    return init_params_list


class embedded_QRS_model1():
    def __init__(self, R, user_embedded_vecs, item_embedded_vecs, init_parms, train_steps):
        self.R = R
        self.user_embedded_vecs = user_embedded_vecs
        self.item_embedded_vecs = item_embedded_vecs
        self.normalize_embdded_vecotrs()
        # plotting the embedded quantum states:
        embedded_state = []
        for user in range(defines._NUM_OF_USERS):
            probs = embedded_QRS_circ_embedding(self.user_embedded_vecs[user])
            embedded_state.append(probs)
        visualiser.plot_embedded_vecs(embedded_state)
        # visualiser.plot_embedded_vecs_3d(embedded_state)

        self.interacted_items_matrix = self.create_interacted_items_matrix()
        self.bad_interacted_items_matrix = self.create_bad_interacted_items_matrix()
        self.uninteracted_items_matrix = self.create_uninteracted_items_matrix()
        self.expected_probs_vecs = self.create_expected_probs_vecs()

        self.avg_num_of_interactions = 0
        for user in range(defines._NUM_OF_USERS):
            self.avg_num_of_interactions += len(self.interacted_items_matrix[user])
        self.avg_num_of_interactions /= defines._NUM_OF_USERS

        print(self.avg_num_of_interactions)
        self.params = np.array(init_parms, requires_grad=True)
        self.train_steps = train_steps
        self.error_ver_weights = self.get_error_vec_weights()
        self.common_item_vec = self.calc_common_items_vecs()
        self.error_vec_weights_for_uninter = self.get_error_vec_weights_for_uninteracted_itesm()
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


    def train(self):
        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.5, beta2=0.599, eps=1e-08)
        start_time_train = time.time()
        print("\n------- TRAINING EMBEDDED ITEM RECOMMENDATION SYS -------")

        for i in range(self.train_steps):
            print("training step:", i)
            self.params = opt_item_item.step(lambda v: self.total_cost_embedded_QRS(v), self.params)

        print("--- embedding train took %s seconds ---" % math.ceil(time.time() - start_time_train))
        visualiser.plot_cost_arrs([self.total_cost])
        # visualiser.plot_cost_arrs(self.error_per_user)


    def total_cost_embedded_QRS(self, params):
        total_cost = 0
        loss = 0
        for user in list(range(defines._NUM_OF_USERS)):
            embedded_vec_for_user = self.user_embedded_vecs[user]

            # running the circuit
            probs = embedded_QRS_circ_reco(embedded_vec_for_user, params)

            # getting expected probs for user
            expected_probs = self.expected_probs_vecs[user]

            # calc the error for the user
            error_per_item = (expected_probs - probs)

            uninteracted_items = self.uninteracted_items_matrix[user]
            bad_interacted_items = self.bad_interacted_items_matrix[user]
            interacted_items = self.interacted_items_matrix[user]

            # remove error on uninteracted items - punishing only on interacted ones
            if len(uninteracted_items) != 0:
                error_per_item._value[uninteracted_items] = 0 # this is an autoguard object - accessing to its values

            # "relu" activation
            # error_per_item_negative = error_per_item/2
            # error_per_item = np.maximum(error_per_item_negative, error_per_item)

            # add punishment to bad interacted items

            error_per_item._value[bad_interacted_items] = error_per_item._value[bad_interacted_items]*10
            error_per_item._value[interacted_items] = error_per_item._value[interacted_items]*2
            error_per_item = error_per_item**2

            error_per_item = error_per_item * self.error_ver_weights
            error_per_item = error_per_item * self.common_item_vec

            cost_for_user = sum(error_per_item)*(len(interacted_items)/self.avg_num_of_interactions)
            self.error_per_user[user].append(cost_for_user._value)

            total_cost += cost_for_user

            # DEBUG:
            # if user == 0:
            #     print("user:", user)
            #     visualiser.print_colored_matrix(expected_probs, [bad_interacted_items, interacted_items], is_vec=1,
            #                                     all_positive=1, digits_after_point=2)
            #     visualiser.print_colored_matrix(probs._value, [bad_interacted_items, interacted_items], is_vec=1,
            #                                     all_positive=1, digits_after_point=2)
            #     # visualiser.print_colored_matrix(self.error_ver_weights, [bad_interacted_items, interacted_items], is_vec=1,
            #     #                                 all_positive=1, digits_after_point=2)
            #     # visualiser.print_colored_matrix(self.error_vec_weights_for_uninter, [bad_interacted_items, interacted_items], is_vec=1,
            #     #                                 all_positive=1, digits_after_point=2)
            #     # visualiser.print_colored_matrix(self.common_item_vec, [bad_interacted_items, interacted_items], is_vec=1,
            #     #                                 all_positive=1, digits_after_point=2)
            #     visualiser.print_colored_matrix(error_per_item._value, [bad_interacted_items, interacted_items], is_vec=1,
            #                                     all_positive=1, digits_after_point=2)
            #     print("cost_for_user", cost_for_user._value, "\n")

        print("total_cost:", total_cost._value)
        print("------------------------------------","\n")
        self.total_cost.append(total_cost._value)
        return total_cost

    # input: list of vectors
    # output: list of vectors which all positive, and the sum of each vector is smaller than pi
    def normalize_embdded_vecotrs(self):
        columns_mins = self.user_embedded_vecs.min(axis=0)
        self.user_embedded_vecs -= columns_mins     # min in evey columns is 0

        global_max = self.user_embedded_vecs.max()+0.0001
        self.user_embedded_vecs /= global_max  # max in all data is 1

        self.user_embedded_vecs *= (2*math.pi/defines._EMBEDDING_SIZE)       # sum of each row is up to pi

        # self.item_embedded_vecs -= self.item_embedded_vecs.min(axis=0)  # min in evey columns is 0
        # self.item_embedded_vecs /= (self.item_embedded_vecs.max() + 0.0001)  # max in all data is 1
        # self.item_embedded_vecs *= (math.pi / defines._EMBEDDING_SIZE)  # sum of each row is up to pi

    def get_recommendation(self, user, uninteracted_items, removed_movie):
        # get the probs vector for user
        embedded_vec_for_user = self.user_embedded_vecs[user]
        probs = embedded_QRS_circ_reco(embedded_vec_for_user, self.params)

        # DEBUG
        print("recommendation for user wo hist removal:", user)
        interacted_items = self.interacted_items_matrix[user]
        bad_interacted_items = self.bad_interacted_items_matrix[user]
        visualiser.print_colored_matrix(probs, [bad_interacted_items, interacted_items, np.array([removed_movie])], is_vec=1,
                                        all_positive=1, digits_after_point=2)

        # history removal - remove interacted items
        # probs[interacted_items] = 0
        # probs[bad_interacted_items] = 0
        # probs = probs/sum(probs)
        # print("recommendation for user w hist removal:", user)
        # visualiser.print_colored_matrix(probs, [bad_interacted_items, interacted_items, np.array([removed_movie])], is_vec=1,
        #                                 all_positive=1, digits_after_point=2)

        return probs

    def get_QRS_reco_matrix(self):
        QRS_reco_matrix = []
        for user in range(defines._NUM_OF_USERS):
            embedded_vec_for_user = self.user_embedded_vecs[user]
            probs = embedded_QRS_circ_reco(embedded_vec_for_user, self.params)
            QRS_reco_matrix.append(probs)
        return QRS_reco_matrix

    def get_QRS_opt_params(self):
        return self.params


    # return vec - where:
    #     items that consumed many times - will have smaller value
    #     items that consumed less times - will have bigger value
    # this protects popular items to get the prop to them
    def get_error_vec_weights(self):
        error_vec_weights = np.zeros(defines._NUM_OF_ITEMS)
        for user in range(defines._NUM_OF_USERS):
            interacted_items = np.where(self.R[user] == 1)[0]
            for i in interacted_items:
                error_vec_weights[i] += 1
        error_vec_weights = [1/i if i>0 else 0 for i in error_vec_weights]
        return error_vec_weights

    # return vec - where:
    #     items that consumed many times - will have smaller value
    #     items that consumed less times - will have bigger value
    # this protects popular items to get the prop to them
    def get_error_vec_weights_for_uninteracted_itesm(self):
        error_vec_weights = np.zeros(defines._NUM_OF_ITEMS)
        for user in range(defines._NUM_OF_USERS):
            interacted_items = np.where(self.R[user] == 1)[0]
            for i in interacted_items:
                error_vec_weights[i] += 1
        max_val = max(error_vec_weights)
        error_vec_weights = [max_val/i if i>0 else max_val for i in error_vec_weights]
        return error_vec_weights



    # calculating for every item - with how many items it comes on average
    # e.g. item 'i' comes always with 3 other items
    # normalizing it to 1
    # items that come as part of a big group we will punish more - because it's expected probs is smaller to begin with
    # logic here is that items that consumed with many other items - then it's a must item (like sequel)
    # items that come as part of a small group - we will punish less - its expected will be larger
    # logic here is the there isn't strong basis for this item - maybe user which watch random staff takes it
    def calc_common_items_vecs(self):
        common_item_vec = np.zeros((defines._NUM_OF_ITEMS), requires_grad=False)
        item_interacted_count = np.zeros((defines._NUM_OF_ITEMS), requires_grad=False)
        for user in range(defines._NUM_OF_USERS):
            for item in range(defines._NUM_OF_ITEMS):
                if self.R[user][item] == 1:
                    item_interacted_count[item] += 1 # number if interatcions marked for this item
                    common_item_vec[item] += np.count_nonzero(self.R[user] == 1) - 1 # adding the other interactions for user

        common_item_vec = [item/count if count > 0 else 0 for (item,count) in zip (common_item_vec, item_interacted_count) ] # calc average item interactions
        common_item_vec = common_item_vec/max(common_item_vec) + 0.1
        return common_item_vec

