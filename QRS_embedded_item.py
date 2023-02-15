import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.templates.layers import BasicEntanglerLayers
from pennylane import numpy as np
from basic_QRS_class import basic_QRS_circ
from basic_QRS_class import basic_QRS

import defines
import visualiser
import math



n_wires = (int(math.log(defines._NUM_OF_ITEMS,2)))
wires_list = list(range(n_wires))
weights_users = np.zeros((1, n_wires), requires_grad=False)


dev_QRS_embedded_layer_item_layer_circ = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_QRS_embedded_layer_item_layer_circ)
def QRS_embedded_layer_item_layer_circ(embedded_params, params):
    # for wire in wires_list:
    #     qml.Hadamard(wire)
    # for p in embedded_params:
    #     p = np.array([p])
    #     AngleEmbedding(p, wires=wires_list, rotation='Z')

    # BasicEntanglerLayers(weights_users, wires=wires_list)
    for wire in wires_list:
        qml.Hadamard(wire)

    for p in params:
        StronglyEntanglingLayers(p, wires=wires_list)

    return qml.probs(wires=wires_list)




# Model Arch: layer of embedded user parasm (rotating only in Z)
# followed by item params, training only the item params one after the other.
class QRS_embedded_layer_item_layer(basic_QRS):
    def __init__(self, R, embedded_users_vec):
        basic_QRS.__init__(self, R)
        self.user_params = embedded_users_vec
        self.normalize_embdded_vecotrs()
        self.history_removal_trained = False

        self.item_params_layers = 5
        self.item_params = {}
        for item in range(defines._NUM_OF_ITEMS):
            self.item_params[item] = self.randomize_init_params_QRS(self.item_params_layers)

        self.history_params_layers = 3
        self.hist_removal_params = {}
        for user in range(defines._NUM_OF_USERS):
            self.hist_removal_params[user] = {}
            for item in self.interacted_items_matrix[user]:
                self.hist_removal_params[user][item] = self.randomize_init_params_QRS(self.history_params_layers)


    # ----------------------------------------------------- TRAIN ------------------------------------------------------
    def train(self):
        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.9, beta2=0.999, eps=1e-08)
        print("\n------- TRAINING RECOMMENDATION SYS -------")
        for epoch in range(5):
            self.total_cost.append(0)
            for user in range(defines._NUM_OF_USERS):
                user_params = self.construct_param_list(np.array(self.user_params[user]), False)
                expected_probs_vec = self.expected_probs_vecs[user]
                cost_ampl_mask = self.cost_ampl_mask_matrix[user]
                for user_epoch in range(5):
                    self.error_per_user[user].append(0)
                    for item in self.interacted_items_matrix[user]:
                        for item_epoch in range(1):

                            # create list of 3D tensors - each tensor is set of parameters
                            item_params = self.construct_param_list(self.item_params[item], True)
                            # print("BEFORE:")
                            # print(self.item_params[item])
                            # print("construnct item_params", item_params, type(item_params))
                            item_params = [opt_item_item.step(
                                self.total_cost_basic_QRS_user_items, *item_params, user=user, expected_probs=expected_probs_vec, cost_ampl_mask=cost_ampl_mask, user_embedded_params=user_params)]
                            # print("AFTER:")
                            # print("construnct item_params", item_params, type(item_params))
                            self.update_params(item, user, item_params, 'item')
                            # print(self.item_params[item])
            print("")
            print(f"total cost: {self.total_cost[-1]:.3f}\n")


        visualiser.plot_cost_arrs([self.total_cost])
        visualiser.plot_cost_arrs(self.error_per_user)


    # ----------------------------------------------------- TRAIN ------------------------------------------------------
    def train_hist_removal(self):
        self.history_removal_trained = True
        print("\n------- TRAINING HISTORY REMOVAL -------")

        cost_ampl_mask = np.ones((defines._NUM_OF_ITEMS), requires_grad=False)
        for user in range(defines._NUM_OF_USERS):
            print("Training user:", user)
            user_params = self.construct_param_list(np.array(self.user_params[user]), False)
            for item in self.interacted_items_matrix[user]:
                opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.9, beta2=0.999, eps=1e-08)

                # creating expected probs vec
                item_params = self.construct_param_list(self.item_params[item], False)
                expected_probs = QRS_embedded_layer_item_layer_circ(user_params, item_params)
                expected_probs[self.interacted_items_matrix[user]] = 0
                expected_probs[self.bad_interacted_items_matrix[user]] = 0
                expected_probs = expected_probs / sum(expected_probs)

                # creating list of 3D tensors - each tensor is set of parameters
                item_params = self.construct_param_list(self.item_params[item], False,
                                                        self.hist_removal_params[user][item], True)
                # print("BEFORE:")
                # print(self.hist_removal_params[user][item])
                # print("construnct item_params", item_params, type(item_params))
                for i in range(20):
                    item_params = opt_item_item.step(
                        self.total_cost_basic_QRS_user_items, *item_params, user=user, expected_probs=expected_probs, cost_ampl_mask=cost_ampl_mask, user_embedded_params=user_params)
                self.update_params(item, user, item_params, 'item', 'hist')

                # print("AFTER:")
                # print("construnct item_params", item_params, type(item_params))
                # print(self.hist_removal_params[user][item])
            print("")




    # calculating the cost on the basic_QRS_cric
    # getting as inputs:  *params = pointer to list of parameters and user=user, item=item expected_probs=expected_probs
    # returns optimized params according to requires_grad field
    # example to a call:
    # params = opt_item_item.step(self.total_cost_basic_QRS_user_items, *params, user=user, item=item, expected_probs=expected_probs)
    def total_cost_basic_QRS_user_items(self, *params, **kwargs):

        probs = QRS_embedded_layer_item_layer_circ(kwargs['user_embedded_params'], params)

        cost_for_user = sum(((kwargs['expected_probs'] - probs) ** 2)*(kwargs['cost_ampl_mask']))

        # errors arnt relevant to hist removal
        if self.history_removal_trained == 0:
            self.error_per_user[kwargs['user']][-1] += cost_for_user._value
            self.total_cost[-1] += cost_for_user._value

        # debug
        # if(kwargs['user']==0):
        #     print("probs\n", probs._value)
        #     print("expected_probs\n", kwargs['expected_probs'])
        return cost_for_user




    # input: list contains: params1, req_grad1, params2 , req_grad2 ....
    def construct_param_list(self, *params_and_gard):
        params_list = []
        for params, req_grad in zip(params_and_gard[::2], params_and_gard[1::2]):
            t = params.copy()
            t.requires_grad = req_grad
            params_list.append(t)
        return params_list


    def update_params(self, item, user, params_list, *params_type_list):
        for i, params_type in enumerate(params_type_list):
            if params_type == 'item':
                self.item_params[item] = params_list[i]
            if params_type == 'hist':
                self.hist_removal_params[user][item] = params_list[i]




    def get_recommendation(self, user, uninteracted_items, removed_movie, debug=1):

        overall_probs = [0]*defines._NUM_OF_ITEMS
        user_params = self.construct_param_list(np.array(self.user_params[user]), False)
        for item in self.interacted_items_matrix[user]:
            if self.history_removal_trained == False:
                item_params = self.construct_param_list(self.item_params[item], False)
            else:
                item_params = self.construct_param_list(self.item_params[item], False,
                                                        self.hist_removal_params[user][item], False)
            overall_probs += QRS_embedded_layer_item_layer_circ(user_params, item_params)

        probs = overall_probs/sum(overall_probs)
        # DEBUG
        if debug:
            print("recommendation for user:", user)
            interacted_items = self.interacted_items_matrix[user]
            bad_interacted_items = self.bad_interacted_items_matrix[user]
            visualiser.print_colored_matrix(probs, [bad_interacted_items, interacted_items, np.array([removed_movie])],
                                            is_vec=1,
                                            all_positive=1, digits_after_point=2)
        return probs

    # input: list of vectors
    # output: list of vectors which all positive, and the sum of each vector is smaller than pi
    def normalize_embdded_vecotrs(self):
        columns_mins = self.user_params.min(axis=0)
        self.user_params -= columns_mins     # min in evey columns is 0

        global_max = self.user_params.max()+0.0001
        self.user_params /= global_max  # max in all data is 1

        self.user_params *= (2*math.pi/defines._EMBEDDING_SIZE)       # sum of each row is up to pi

        # self.item_embedded_vecs -= self.item_embedded_vecs.min(axis=0)  # min in evey columns is 0
        # self.item_embedded_vecs /= (self.item_embedded_vecs.max() + 0.0001)  # max in all data is 1
        # self.item_embedded_vecs *= (math.pi / defines._EMBEDDING_SIZE)  # sum of each row is up to pi
