import numpy as np
import collections
import random
import pennylane as qml
import torch
import tqdm


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    # 添加
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 抽样
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

    # 量子退火
    def replace_experience(self, temperature):
        # 计算经验的"贡献"值：这里假设贡献值是reward的加权
        reward = np.array([reward for _, _, reward, _, _ in self.buffer])

        # 将经验贡献作为输入，进行量子退火选择
        selected_experience = quantum_annealing_experience_selection(self.buffer, temperature, reward)
        return selected_experience


# 量子退火
def quantum_annealing_experience_selection(buffer, temperature, reward):
    # 设置固定数量的量子比特
    num_qubits = 3  # 固定为3个量子比特
    dev = qml.device("default.qubit", wires=num_qubits)

    # 定义量子电路
    @qml.qnode(dev)
    def quantum_circuit():
        # 初始化所有量子比特为均匀叠加态
        for i in range(num_qubits):
            qml.Hadamard(wires=i)

        # 计算贡献的哈密顿量（目标函数）
        # 这里我们可以根据贡献调整量子比特的权重
        for i in range(num_qubits):
            qml.RZ((max(reward[i])+min(reward[i]))/2 * temperature, wires=i)  # 温度参数影响贡献的值
        # print(f"reward[i]: {reward[i]}, temperature: {temperature}, reward[i] * temperature: {reward[i] * temperature}")


        # 测量量子比特的期望值
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    # 执行量子电路
    result = quantum_circuit()

    # 选择具有较高贡献的经验
    selected_indices = np.argsort(result)[-len(buffer) // 2:]  # 选择贡献最大的经验
    selected_experience = [buffer[i] for i in selected_indices]

    return selected_experience


# 数据平滑处理，查看数据趋势，减少不必要噪声，好像只在train.py里面画图的地方调用过一次
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# 训练智能体，根据当前状态选择下一状态，重复10回合，但是这个函数似乎没有被调用
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, temperature=1.0):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)  # 计算并添加贡献度
                    state = next_state
                    episode_return += reward

                    # 经验池达到一定大小后，进行量子退火优化
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                        replay_buffer.replace_experience(temperature)  # 使用量子退火替换经验

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def update_temperature(temperature, decay_factor=0.99):
    """ 随着训练的进行，温度逐渐降低 """
    return temperature * decay_factor
