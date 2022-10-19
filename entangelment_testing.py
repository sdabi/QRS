import defines
import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane import numpy as np

n_wires = defines._NUM_OF_QUBITS_TO_OPTIMIZE
wires = list(range(n_wires))


dev_entangling_tester = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_entangling_tester)
def entangling_tester_circ(params):
    for wire in wires:
        qml.Hadamard(wire)
    StronglyEntanglingLayers(params, wires=wires)
    return qml.probs(wires=wires)

class entangling_tester():
    def __init__(self, goal, train_steps=10, layers=1):
        shape = StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n_wires)
        self.params = np.random.random(size=shape)
        self.goal = goal
        self.train_steps = train_steps
        self.total_cost = []


    def train(self):
        opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.5, beta2=0.599, eps=1e-08)

        for i in range(self.train_steps):
            self.params = opt_item_item.step(lambda v: self.total_cost_embedded_QRS(v), self.params)


    def total_cost_embedded_QRS(self, params):
        probs = entangling_tester_circ(params)
        error = sum((self.goal - probs)**2)
        return error

    def test_entangling(self):
        probs = entangling_tester_circ(self.params)
        error = sum((self.goal - probs)**2)
        return error

    def print_probs(self):
        print(entangling_tester_circ(self.params))
