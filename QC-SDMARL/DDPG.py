import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import rl_utils
import tool_functions
from misc import categorical_sample
import pennylane as qml

# class TwoLayerFC(torch.nn.Module):
#     def __init__(self, num_in, num_out, hidden_dim):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(num_in, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = torch.nn.Linear(hidden_dim, num_out)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = torch.tanh(self.fc2(x))
#         return self.fc3(x)

# Actor网络
class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x, flag = True,
                return_log_pi=True):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if flag :
            return x
        else:
            probs = F.softmax(x,dim=1)
            on_gpu = next(self.parameters()).is_cuda
            int_act,act = categorical_sample(probs,use_cuda = on_gpu)
            rets = []
            if return_log_pi :
                log_probs = F.log_softmax(x, dim=1)

            if return_log_pi:
                # return log probability of selected action
                rets.append(log_probs.gather(1, int_act))
            return rets


# Critic网络
class Critic(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim, agent_num):
        super().__init__()
        # 压缩其他智能体信息
        # self.fc1 = torch.nn.Linear(num_in * (agent_num - 2) / agent_num, 64)
        x1 = num_in * (agent_num - 2) / agent_num
        x1 = int(x1)
        self.fc1 = torch.nn.Linear(x1, 32)
        # self.fc2 = torch.nn.Linear(64, 8)
        x2 = (num_in / agent_num) * 2 + 32
        x2 = int(x2)
        # 将压缩后的与自己与最优智能体的信息拼接输入
        self.fc3 = torch.nn.Linear(x2, hidden_dim)
        # self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, *args):
        # 如果当前智能体是最优智能体
        # 打算扩大自己信息的影响，将自己的信息扩大
        if args[2][1]:
            # print("当前更新智能体的智能体是最优智能体")
            x = random.randint(0, len(args[0]))
            # 生成一个不等于最优智能体的信息的数
            while x == args[2][2] or x > (len(args[0])-1):
                x = random.randint(0, len(args[0]) - 1)
            array_obs = []
            array_action = []
            # 记录优势智能体的信息
            print("x为:", x)
            ad_obs = torch.cat((args[0][args[2][0]], args[0][x]), dim=1)
            ad_act = torch.cat((args[1][args[2][0]], args[1][x]), dim=1)
            # 将优势智能体的信息与自己的信息拼接起来
            advan = torch.cat((ad_obs, ad_act), dim=1)
            # 将其它智能体的信息提取到中间数组里
            for i in range(len(args[1])):
                if (i == args[2][0]) | (i == x):
                    continue
                array_obs.append(args[0][i])
                array_action.append(args[1][i])
            # 将其他智能体的信息拼接起来
            disadvan_obs = array_obs[0]
            disadvan_action = array_action[0]
            for i in range(len(array_obs) - 1):
                disadvan_obs = torch.cat((disadvan_obs, array_obs[i + 1]), dim=1)
                disadvan_action = torch.cat((disadvan_action, array_action[i + 1]), dim=1)
            dis_input = torch.cat((disadvan_obs, disadvan_action), dim=1)
            # 将其它智能体的信息输入到网络中进行压缩
            dis_input = F.relu(self.fc1(dis_input))
            # dis_input = F.relu(self.fc2(dis_input))

            # 将压缩后的信息与优势智能体的信息拼接起来
            total_input = torch.cat((advan, dis_input), dim=1)

            total_input1 = F.relu(self.fc3(total_input))
            # total_input = F.relu(self.fc4(total_input))
            return self.fc5(total_input1)

        # 如果当前智能体不是最优智能体
        else:
            # print("当前更新智能体的智能体不是最优智能体")
            array_obs = []
            array_action = []
            # 记录优势智能体的信息
            ad_obs = torch.cat((args[0][args[2][0]], args[0][args[2][2]]), dim=1)
            ad_act = torch.cat((args[1][args[2][0]], args[1][args[2][2]]), dim=1)
            # 将优势智能体的信息与自己的信息拼接起来
            advan = torch.cat((ad_obs, ad_act), dim=1)
            # 将其它智能体的信息提取到中间数组里
            for i in range(len(args[1])):
                if (i == args[2][0]) | (i == args[2][2]):
                    continue
                array_obs.append(args[0][i])
                array_action.append(args[1][i])
            # 将其他智能体的信息拼接起来
            disadvan_obs = array_obs[0]
            disadvan_action = array_action[0]
            for i in range(len(array_obs) - 1):
                disadvan_obs = torch.cat((disadvan_obs, array_obs[i + 1]), dim=1)
                disadvan_action = torch.cat((disadvan_action, array_action[i + 1]), dim=1)
            dis_input = torch.cat((disadvan_obs, disadvan_action), dim=1)
            # 将其它智能体的信息输入到网络中进行压缩
            dis_input = F.relu(self.fc1(dis_input))
            # dis_input = F.relu(self.fc2(dis_input))

            # 将压缩后的信息与优势智能体的信息拼接起来
            total_input = torch.cat((advan, dis_input), dim=1)

            total_input1 = F.relu(self.fc3(total_input))
            # total_input = F.relu(self.fc4(total_input))
            return self.fc5(total_input1)


class DDPG:
    ''' DDPG算法 '''

    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device, flag, agt_i, agent_num):
        # 训练使用
        if flag:
            self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
            self.target_actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)

            self.critic = Critic(critic_input_dim, 1, hidden_dim, agent_num).to(device)
            self.target_critic = Critic(critic_input_dim, 1, hidden_dim, agent_num).to(device)

            # self.actor.load_state_dict(torch.load('agent%d_actor4.pth' % agt_i))
            # self.critic.load_state_dict(torch.load('agent%d_critic4.pth' % agt_i))

            self.target_critic.load_state_dict(self.critic.state_dict())
            self.target_actor.load_state_dict(self.actor.state_dict())

            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # 测试使用
        else:
            self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
            self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
            self.actor.load_state_dict(torch.load('./epoch5000/agent%d_actor3.pth' % agt_i))
            self.critic.load_state_dict(torch.load('./epoch5000/agent%d_critic3.pth' % agt_i))

    # 做动作
    def take_action(self, state, explore=False):
        action = self.actor(state)
        # 采样
        if explore:
            action = tool_functions.gumbel_softmax(action)
        else:
            action = tool_functions.onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    # 更新目标网络参数
    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
