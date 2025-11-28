import torch
import numpy as np
import pennylane as qml
from MADDPG import MADDPG
from quantum_utils import dev, quantum_circuit  # 导入量子设备和电路
from scipy.optimize import minimize


def post_process_weights(weights, clip_range=(-0.1, 0.1)):
    """
    对权重进行后处理，标准化并限制范围
    """
    mean, std = weights.mean(), weights.std()
    processed_weights = (weights - mean) / (std + 1e-6)  # 标准化
    processed_weights = np.clip(processed_weights, *clip_range)  # 限制范围
    return torch.tensor(processed_weights, dtype=torch.float32)

def quantum_weight_generator(num_qubits, num_weights, max_qubits=10):
    """生成量子权重"""
    quantum_circuit(num_qubits)
    quantum_results = []
    for _ in range(num_weights):
        result = quantum_circuit(num_qubits)
        quantum_results.append(float(result))

    # 后处理权重
    return post_process_weights(np.array(quantum_results))

# hamiltonian = qml.PauliZ(0) + qml.PauliZ(1) + qml.PauliZ(2)
def vqe_weight_optimization(num_qubits,num_weights, dev):
    """
    使用 VQE 算法优化权重
    """

    @qml.qnode(dev)
    def circuit(params):
        # 应用参数化的量子门
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
            qml.RY(params[i], wires=i)

        # 应用额外的量子门以模拟更复杂的电路
        # for i in range(3):
        #     qml.CNOT(wires=[(i, i + 1) if i < 2 else (i, 0)])
        #     qml.RY(params[3 + i], wires=i)

        return qml.expval(qml.PauliZ(0))

    # 定义优化的目标函数
    def cost_fn(params):
        return circuit(params)

    # 初始参数
    init_params = np.random.uniform(0, 2 * np.pi, num_weights)

    # 使用 scipy 的 minimize 函数进行优化
    result = minimize(cost_fn, init_params, method='COBYLA')

    # 返回优化后的权重
    return result.x


# 调用 VQE 函数
class QMADPG(MADDPG):
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims, critic_input_dim,
                 gamma, tau, agent_num, num_qubits=3, dynamic_update_freq=5):
        """
        初始化 QMADDPG
        """
        super().__init__(env, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims, critic_input_dim,
                         gamma, tau, agent_num)
        self.num_qubits = num_qubits
        self.dynamic_update_freq = dynamic_update_freq

        for agent in self.agents:
            num_weights = state_dims[0] * hidden_dim
            quantum_weights = quantum_weight_generator(self.num_qubits, num_weights)

            if quantum_weights is not None:
                # 使用VQE算法优化权重
                optimized_weights = vqe_weight_optimization(num_qubits,num_weights, dev)

                # 将NumPy数组转换为PyTorch张量
                optimized_weights_torch = torch.tensor(optimized_weights, dtype=torch.float32)

                # 检查权重数量是否匹配
                expected_size = agent.actor.fc1.weight.shape[0] * agent.actor.fc1.weight.shape[1]
                if optimized_weights_torch.numel() != expected_size:
                    raise ValueError(f"Optimized weights size {optimized_weights_torch.numel()} does not match expected size {expected_size}")

                # 调整优化后的权重形状，使其与fc1的权重形状匹配
                optimized_weights_torch = optimized_weights_torch.view(agent.actor.fc1.weight.shape)

                # 将优化后的权重应用到actor和target_actor
                agent.actor.fc1.weight.data = optimized_weights_torch
                agent.target_actor.fc1.weight.data = optimized_weights_torch
    def dynamic_weight_update(self, agent_idx, state, reward, next_state, global_reward):
        """
        动态更新指定智能体的权重
        """
        agent = self.agents[agent_idx]
        num_weights = state.shape[1] * agent.actor.fc1.out_features

        updated_weights = quantum_weight_generator(self.num_qubits, num_weights)

        if updated_weights is not None:
            optimized_weights = vqe_weight_optimization(self.num_qubits,num_weights, dev)

            # 全局奖励约束权重调整
            adjustment_factor = min(1.0, global_reward / 10.0)  # 限制调整幅度
            agent.actor.fc1.weight.data = (
                0.99 * agent.actor.fc1.weight.data
                + adjustment_factor * optimized_weights.view_as(agent.actor.fc1.weight)
            )
            agent.target_actor.fc1.weight.data = (
                0.9 * agent.target_actor.fc1.weight.data
                + adjustment_factor * optimized_weights.view_as(agent.target_actor.fc1.weight)
            )

    def update(self, sample, i_agent, max_reward, i_episode):
        """
        重载更新方法
        """
        super().update(sample, i_agent, max_reward, i_episode)
        if i_episode % self.dynamic_update_freq == 0:  # 控制动态更新频率
            obs, act, rew, next_obs, _ = sample
            global_reward = max_reward  # 使用全局奖励作为约束
            self.dynamic_weight_update(i_agent, obs[i_agent], rew[i_agent], next_obs[i_agent], global_reward)