# quantum_utils.py
import pennylane as qml
import numpy as np
num_qubits=3

def create_quantum_device(num_qubits):
    max_qubits = 10  
    num_qubits = min(num_qubits, max_qubits) 
    return qml.device("default.qubit", wires=num_qubits)


dev = create_quantum_device(num_qubits)

@qml.qnode(dev)
def quantum_circuit(num_qubits):

    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        qml.RX(np.random.uniform(0, 2 * np.pi), wires=i)
    return qml.expval(qml.PauliX(0))

