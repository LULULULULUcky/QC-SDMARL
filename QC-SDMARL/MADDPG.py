import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import rl_utils
import tool_functions
import DDPG as DP


class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims, critic_input_dim, gamma,
                 tau, agent_num=12):
        self.agents = []
        for i in range(len(env.agents)):
            self.agents.append(
                DP.DDPG(state_dims[i], action_dims[i], critic_input_dim, hidden_dim, actor_lr, critic_lr, device, True,
                        i, agent_num))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        # 用来判断是不是最优智能体
        self.flag = [0, False, 0]
        self.reward_scale = 100

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, env, states, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(len(env.agents))
        ]
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    # 更新参数
    def update(self, sample, i_agent, max_reward, i_episode):
        x = []
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]
        self.flag[0] = i_agent
        self.flag[2] = max_reward
        # 提出的更新参数的方式
        # 当前智能体判断自己是不是最优智能体
        # 如果是最优智能体，保持原有的更新方式
        if i_agent == max_reward:
            self.flag[1] = True
        else:
            self.flag[1] = False
        self.agents[0].critic_optimizer.zero_grad()
        # all_target_act = [tool_functions.onehot_from_logits(pi(_next_obs)) for pi, _next_obs in
        #                   zip(self.target_policies, next_obs)]
        all_target_act = [pi(_next_obs, flag=False) for pi, _next_obs in
                          zip(self.target_policies, next_obs)]
        all_target_act1 = [tool_functions.onehot_from_logits(pi(_next_obs, flag=True)) for pi, _next_obs in
                           zip(self.target_policies, next_obs)]
        # target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        # 使用奖励来指导更新参数
        # target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * cur_agent.target_critic(target_critic_input,
        #                                                                                       self.flag) * (
        #                               1 - done[i_agent].view(-1, 1))

        x.append(next_obs)
        x.append(all_target_act1)
        x.append(self.flag)
        # if self.reward_scale:
        #     self.reward_scale = 100 - i_episode * 0.1
        # if i_episode < 100:
        #     target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * (self.agents[0].target_critic(*x) * (
        #             1 - done[i_agent].view(-1, 1)) - torch.sum(all_target_act[i_agent][0]).float() / self.reward_scale)

        target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * (self.agents[0].target_critic(*x) *
                                                                           1 - done[i_agent].view(-1, 1))
        # critic_input = torch.cat((*obs, *act), dim=1)
        x = [obs, act, self.flag]

        critic_value = self.agents[0].critic(*x)
        x = []
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()
        self.agents[0].critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent], flag=True)
        cur_act_vf_in = tool_functions.gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(tool_functions.onehot_from_logits(pi(_obs, flag=True)))
        # vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        x = [obs, all_actor_acs, self.flag]
        actor_loss = -self.agents[0].critic(*x).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)

    def save(self, agt_i):
        torch.save(self.agents[agt_i].actor.state_dict(), 'agent%s_actor4.pth' % agt_i)
        torch.save(self.agents[agt_i].critic.state_dict(), 'agent%s_critic4.pth' % agt_i)
