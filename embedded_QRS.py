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
n_user_wires = 0
n_wires = n_user_wires + n_item_wires

item_wires = list(range(n_item_wires))
user_wires = list(range(n_item_wires, n_wires))
wires_list = item_wires + user_wires

# weights =  np.random.random((1, n_wires))
weights_users = np.zeros((1, n_item_wires), requires_grad=False)
weights_all = np.zeros((1, n_wires), requires_grad=False)



def density_matrix(state):
    return state * np.conj(state).T





# wrap device in qml.qnode
dev_embedded_ItemItem = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_embedded_ItemItem)
# def circuit(params,state=None):
def embedded_QRS_circ(embedded_params, params, expected_state_dm):
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

    return qml.expval(qml.Hermitian(expected_state_dm, wires=item_wires))


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


class embedded_QRS():
    def __init__(self, R, user_embedded_vecs, item_embedded_vecs, train_steps):
        self.R = R
        self.user_embedded_vecs = user_embedded_vecs
        self.item_embedded_vecs = item_embedded_vecs
        self.normalize_embdded_vecotrs()
        visualiser.plot_embedded_vecs(self.user_embedded_vecs)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=defines._NUM_OF_LAYERS, n_wires=n_item_wires)
        self.params = np.random.random(size=shape, requires_grad=True)
        self.train_steps = train_steps
        self.error_ver_weights = self.get_error_vec_weights()
        self.total_cost = []
        self.error_per_user = []
        for i in range(defines._NUM_OF_USERS):
            self.error_per_user.append([])

    def train(self):
        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.5, beta2=0.599, eps=1e-08)
        start_time_train = time.time()
        print("\n------- TRAINING EMBEDDED ITEM RECOMMENDATION SYS -------")

        for i in range(self.train_steps):
            if i % 10 == 0:
                print("training step:", i)
                # print(self.params)
            self.params = opt_item_item.step(lambda v: self.total_cost_embedded_QRS(v), self.params)

        print("--- embedding train took %s seconds ---" % math.ceil(time.time() - start_time_train))
        visualiser.plot_cost_arrs([self.total_cost])
        visualiser.plot_cost_arrs(self.error_per_user)


    def total_cost_embedded_QRS(self, params):
        total_cost = 0
        loss = 0
        for user in list(range(defines._NUM_OF_USERS)):
            embedded_vec_for_user = self.user_embedded_vecs[user]

            # running the circuit
            probs = embedded_QRS_circ_reco(embedded_vec_for_user, params)

            # getting the indecies where user have positive interaction
            interacted_items = np.where(self.R[user] == 1)[0]

            # getting the indecies where user have negetive interaction
            bad_interacted_items = np.where(self.R[user] == defines._BAD_SAMPLED_INTER)[0]

            # building the expected prop array
            # for interacted items - the expected val is _MAX_HIST_INTER_WEIGHT/(num of interacted items)
            # for un-interacted items - the expected val is (1-_MAX_HIST_INTER_WEIGHT)/ num of un-interacted items
            # for bad-interacted items - the expected val is 0
            expected_probs = np.ones(defines._NUM_OF_ITEMS, requires_grad=False) * (1 - defines._MAX_HIST_INTER_WEIGHT) / (
                        defines._NUM_OF_ITEMS - len(interacted_items) - len(bad_interacted_items))
            if (len(interacted_items) > 0):
                expected_probs[interacted_items] = defines._MAX_HIST_INTER_WEIGHT / len(interacted_items)
            expected_probs[bad_interacted_items] = 0


            # FIDELITY CIRC
            # expected_probs = np.sqrt(expected_probs)
            # expected_probs = [[np.array((i), requires_grad=False)] for i in expected_probs]
            # expected_output_state_dm = density_matrix(expected_probs)
            # f = embedded_QRS_circ(embedded_vec_for_user, params, expected_output_state_dm)
            # loss = loss + (1 - f) ** 2
            # print("total loss", (loss/defines._NUM_OF_USERS))
            # return loss/defines._NUM_OF_USERS

            uninteracted_items = [i for i in range(defines._NUM_OF_ITEMS) if i not in interacted_items and i not in bad_interacted_items]
            # calc the error for the user
            error_per_item = (probs - expected_probs)**2
            if len(uninteracted_items) != 0:
                error_per_item._value[uninteracted_items] = 0 # this is an autoguard object - accessing to its values

            max_error_for_user = 0
            if len(bad_interacted_items) != 0:
                max_error_for_user = 1 + 1 / len(interacted_items)
            else:
                if len(interacted_items) == 1:
                    max_error_for_user = 1
                else:
                    max_error_for_user = 1 - 1 / len(interacted_items)

            cost_for_user = sum(error_per_item)/max_error_for_user
            self.error_per_user[user].append(cost_for_user._value)

            total_cost += cost_for_user

            # DEBUG:
            if user == 0:
                print("probs for user:")
                visualiser.print_colored_matrix(probs._value, [bad_interacted_items, interacted_items], is_vec=1, all_positive=1, digits_after_point=2)
                print("expected probs for user:")
                visualiser.print_colored_matrix(expected_probs, [bad_interacted_items, interacted_items], is_vec=1, all_positive=1, digits_after_point=2)

        print("total_cost:", total_cost._value,"\n")
        self.total_cost.append(total_cost._value)
        return total_cost

    # input: list of vectors
    # output: list of vectors which all positive, and the sum of each vector is smaller than pi
    def normalize_embdded_vecotrs(self):
        self.user_embedded_vecs -= self.user_embedded_vecs.min(axis=0)     # min in evey columns is 0
        self.user_embedded_vecs /= (self.user_embedded_vecs.max()+0.0001)  # max in all data is 1
        self.user_embedded_vecs *= (math.pi/defines._EMBEDDING_SIZE)       # sum of each row is up to pi

        self.item_embedded_vecs -= self.item_embedded_vecs.min(axis=0)  # min in evey columns is 0
        self.item_embedded_vecs /= (self.item_embedded_vecs.max() + 0.0001)  # max in all data is 1
        self.item_embedded_vecs *= (math.pi / defines._EMBEDDING_SIZE)  # sum of each row is up to pi

    def get_recommendation(self, user, uninteracted_items):
        # getting the indecies where user have positive interaction
        interacted_items = np.where(self.R[user] == 1)[0]

        # getting the indecies where user have negetive interaction
        bad_interacted_items = np.where(self.R[user] == defines._BAD_SAMPLED_INTER)[0]

        embedded_vec_for_user = self.user_embedded_vecs[user]

        # get the probs vector for user
        probs = embedded_QRS_circ_reco(embedded_vec_for_user, self.params)

        print("recommendation for user wo hist removal:", user)
        visualiser.print_colored_matrix(probs, [bad_interacted_items, interacted_items], is_vec=1,
                                        all_positive=1, digits_after_point=2)

        # history removal - remove interacted items
        probs[interacted_items] = 0
        probs[bad_interacted_items] = 0
        probs = probs/sum(probs)

        # print("recommendation for user w hist removal:", user)
        # visualiser.print_colored_matrix(probs, [bad_interacted_items, interacted_items], is_vec=1,
        #                                 all_positive=1, digits_after_point=2)
        print("")  # new line

        return probs



    def get_error_vec_weights(self):
        error_vec_weights = np.zeros(defines._NUM_OF_ITEMS) + 0.001
        for user in range(defines._NUM_OF_USERS):
            interacted_items = np.where(self.R[user] == 1)[0]
            for i in interacted_items:
                error_vec_weights[i] += 1
        print(error_vec_weights)
        error_vec_weights = 1/error_vec_weights
        print(error_vec_weights)
        return error_vec_weights