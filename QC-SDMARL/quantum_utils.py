# quantum_utils.py
import pennylane as qml
import numpy as np
num_qubits=3
# 定义量子设备
def create_quantum_device(num_qubits):
    max_qubits = 10  # 设置最大量子比特数
    num_qubits = min(num_qubits, max_qubits)  # 限制量子比特数量
    return qml.device("default.qubit", wires=num_qubits)

# 创建量子设备实例
dev = create_quantum_device(num_qubits)

# 定义量子电路
@qml.qnode(dev)
def quantum_circuit(num_qubits):
    """量子电路生成权重"""
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        qml.RX(np.random.uniform(0, 2 * np.pi), wires=i)
    return qml.expval(qml.PauliX(0))

