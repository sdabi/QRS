import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.templates.embeddings import AngleEmbedding
from pennylane import numpy as np
import time

import math
import defines
import visualiser


n_wires = (int(math.log(defines._NUM_OF_ITEMS,2)))
wires_list = list(range(n_wires))



# wrap device in qml.qnode
dev_embedded_ItemItem = qml.device('default.qubit', wires=n_wires)

@qml.qnode(dev_embedded_ItemItem)
# def circuit(params,state=None):
def embedded_QRS_circ(embedded_params, params):
    for item_wire in wires_list:
        qml.Hadamard(item_wire)
    AngleEmbedding(embedded_params, wires=wires_list, rotation='Z')
    StronglyEntanglingLayers(params, wires=wires_list)
    return qml.probs(wires=wires_list)


class embedded_QRS():
    def __init__(self, R, user_embedded_vecs, train_steps):
        self.R = R
        self.user_embedded_vecs = user_embedded_vecs
        self.normalize_embdded_vecotrs()
        visualiser.plot_embedded_vecs(self.user_embedded_vecs)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=defines._NUM_OF_LAYERS, n_wires=n_wires)
        self.params = np.random.random(size=shape)
        self.train_steps = train_steps


    def train(self):
        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.3, beta2=0.3, eps=1e-08)
        start_time_train = time.time()
        print("\n------- TRAINING EMBEDDED ITEM RECOMMENDATION SYS -------")

        for i in range(self.train_steps):
            self.params = opt_item_item.step(lambda v: self.total_cost_embedded_QRS(v), self.params)
        print("--- embedding train took %s seconds ---" % math.ceil(time.time() - start_time_train))

    def total_cost_embedded_QRS(self, params):
        total_cost = 0
        for user in list(range(defines._NUM_OF_USERS)):
            embedded_vec_for_user = self.user_embedded_vecs[user]

            # running the circuit
            probs = embedded_QRS_circ(embedded_vec_for_user, params)

            # getting the indecies where user have positive interaction
            interacted_items = np.where(self.R[user] == 1)[0]

            # getting the indecies where user have negetive interaction
            bad_interacted_items = np.where(self.R[user] == defines._BAD_SAMPLED_INTER)[0]

            # building the expected prop array
            # for interacted items - the expected val is _MAX_HIST_INTER_WEIGHT/(num of interacted items)
            # for un-interacted items - the expected val is (1-_MAX_HIST_INTER_WEIGHT)/ num of un-interacted items
            # for bad-interacted items - the expected val is 0
            expected_probs = np.ones(defines._NUM_OF_ITEMS) * (1 - defines._MAX_HIST_INTER_WEIGHT) / (
                        defines._NUM_OF_ITEMS - len(interacted_items) - len(bad_interacted_items))
            if (len(interacted_items) > 0):
                expected_probs[interacted_items] = defines._MAX_HIST_INTER_WEIGHT / len(interacted_items)
            expected_probs[bad_interacted_items] = 0

            uninteracted_items = [i for i in range(defines._NUM_OF_ITEMS) if i not in interacted_items and i not in bad_interacted_items]
            # calc the error for the user
            error_per_item = ((probs - expected_probs) ** 2)
            if len(uninteracted_items) != 0:
                error_per_item._value[uninteracted_items] = 0 # this is an autoguard object - accessing to its values
            cost_for_user = sum(error_per_item)
            total_cost += cost_for_user

            # DEBUG:
            if user == 0:
                print("error for user: ", cost_for_user._value, "\nprobs for user 0:\n", probs._value, "\nexpected probs:\n", expected_probs)

        print("total_cost", total_cost._value)
        return total_cost

    # input: list of vectors
    # output: list of vectors which all positive, and the sum of each vector is smaller than pi
    def normalize_embdded_vecotrs(self):
        self.user_embedded_vecs -= self.user_embedded_vecs.min(axis=0)     # min in evey columns is 0
        self.user_embedded_vecs /= (self.user_embedded_vecs.max()+0.0001)  # max in all data is 1
        self.user_embedded_vecs *= (math.pi/defines._EMBEDDING_SIZE)       # sum of each row is up to pi

    def get_recommendation(self, user, uninteracted_items):
        embedded_vec_for_user = self.user_embedded_vecs[user]
        print("embedded_vec_for_user:", user, "is:\n", embedded_vec_for_user)
        # get the probs vector for user
        probs = embedded_QRS_circ(embedded_vec_for_user, self.params)

        # history removal - remove interacted items
        interacted_items = [i for i in range(defines._NUM_OF_ITEMS) if i not in uninteracted_items]
        probs[interacted_items] = 0

        # normalize
        probs = probs/sum(probs)

        return probs