import time
from multienvironment import Multiagent
from multienvironment import all_entity_positions
from multienvironment import agent_velocity_data
from multienvironment import target_velocity_data
from make_world import MakeWorld
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import rl_utils
from rl_utils import update_temperature
import tool_functions
import DDPG as DP
import QMADDPG as QMADP


num_episodes = 10
episode_length = 600  
buffer_size = 100000
hidden_dim = 32
actor_lr = 3e-2
critic_lr = 3e-2
gamma = 0.95
tau = 1e-2
batch_size =512
agent_num = 6

device = torch.device("cpu")
update_interval = 3000#3000
minimal_size = 4000
n_iterations = 1

temperature = 1.0
decay_factor = 0.9  

num_train_rounds = 1  

for round_num in range(num_train_rounds):
    print(f"Starting training round {round_num + 1}...")

    for count in range(n_iterations):
        world_res = MakeWorld()
        world = world_res.make_world()
        env = Multiagent(world, world_res, MakeWorld.rest_world, MakeWorld.reward, MakeWorld.observation)

        replay_buffer = rl_utils.ReplayBuffer(buffer_size)
        max_agent = []
        state_dims = []
        action_dims = []
        total_step = 0
        return_list = []
        reward_list = []
        reward_t = []


        def evaluate(maddpg, n_episode=10, episode_length=500):
   
            world_res = MakeWorld()
            world = world_res.make_world()
            env = Multiagent(world, world_res, MakeWorld.rest_world, MakeWorld.reward, MakeWorld.observation)
            returns = np.zeros(len(env.agents))
            for _ in range(n_episode):
                obs = env.reset()
                for t_i in range(episode_length):
                    actions = maddpg.take_action(env, obs, explore=False) 
                    obs, rew, done, info = env.step(actions)
                    rew = np.array(rew)
                    returns += rew / n_episode
            return returns.tolist()


        for action_space in env.action_space:
            # Discre(5),n is 5，
            action_dims.append(action_space.n)
        for state_space in env.observation_space:
            state_dims.append(state_space.shape[0])
        critic_input_dim = sum(state_dims) + sum(action_dims)

        maddpg = QMADP.QMADPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims,
                              critic_input_dim, gamma, tau, agent_num, num_qubits=3)

      
        energy_records = []
        for i_episode in range(num_episodes):
            state = env.reset()
            episode_energy = []  

            #     reward_list.append(np.sum(reward_t))
            if len(reward_t) > 0:
                reward_list.append(reward_t.copy())
            else:
                reward_list.append(np.zeros(agent_num)) 

            print(np.sum(reward_t))
            # print(reward_list[len(reward_list) - 1])
            print(i_episode)


            reward_t = np.zeros(agent_num)

            for e_i in range(episode_length):
        
                actions = maddpg.take_action(env, state, explore=True)
                next_state, reward, done, _ = env.step(actions)

                reward = np.array(reward)
                reward_t += reward

                replay_buffer.add(state, actions, reward, next_state, done)

                state = next_state
                total_step += 1


                current_energy = [agent.energy for agent in env.agents]
                episode_energy.append(current_energy)

                if replay_buffer.size() >= minimal_size and total_step % update_interval == 0:
           
                    sample = replay_buffer.sample(batch_size)

                    for i in range(2):
                        if i == 0:
              
                            def stack_array(x):
                                rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
                                return [torch.FloatTensor(np.vstack(aa)).to(device) for aa in rearranged]


                            if len(reward_t):
                        
                                max_element = np.argmax(reward_t)
                                max_agent.append(max_element)
                            sample1 = [stack_array(x) for x in sample]

                    
                            for a_i in range(len(env.agents)):
                                maddpg.update(sample1, a_i, max_element, i_episode)

              
                            maddpg.update_all_targets()

                        else:
                      
                            flag = list(sample)
                            flag[2] = list(flag[2])
                            x = [np.sum(flag[2][i]) for i in range(len(flag[2]))]
                            x = np.array(x)
                            for i in range(len(flag)):
                                flag[i] = np.array(list(flag[i]))
                                flag[i] = flag[i][np.argsort(x)[:256]]  

                            sample = tuple(flag)

                            sample1 = [stack_array(x) for x in sample]
                            for a_i in range(len(env.agents)):
                                maddpg.update(sample1, a_i, max_element, i_episode)

                
                            maddpg.update_all_targets()

             
                    replay_buffer.replace_experience(temperature)  
      
                temperature = update_temperature(temperature, decay_factor) 


            energy_records.append(episode_energy)

            if i_episode == num_episodes-1:
                for a_i in range(len(env.agents)):
                    maddpg.save(a_i)

        # for i, item in enumerate(reward_list):
        #     print(f"Index {i}, Length {len(item)}, Type {type(item)}, First element type {type(item[0]) if len(item) > 0 else 'Empty'}")

        # print(reward_list)
        # print(reward_list)
        return_array = np.array(reward_list)
        # np.save(file="./sanwei/newsample_756.npy", arr=return_array)
        # max_array = np.array(max_agent)
        # np.save(file="./sanwei/max_sample_756.npy", arr=max_array)
        #
        # np.save('sanwei/12-4-env/QC-SDMARL_all_entity_positions.npy', np.array(all_entity_positions))
        # np.save('sanwei/12-4-env/QC-SDMARL_agent_velocity_data1.npy', np.array(agent_velocity_data))
        # np.save('sanwei/12-4-env/QC-SDMARL_target_velocity_data.npy', np.array(target_velocity_data))
        #

        # remaining_energy = [agent.energy for agent in env.agents]
        # np.save(file="sanwei/12-4-env/QC-SDMARL_remaining_energy.npy", arr=np.array(remaining_energy))
        #

        # np.save(file="sanwei/12-4-env/QC-SDMARL_energy_records.npy", arr=np.array(energy_records))
        #
        # np.save(f"./sanwei/4追2-10/reward-Our-0724{count+0}.npy", return_array)

        total_rewards = np.sum(return_array, axis=1)
        np.save(f"sanwei/9-3/QC_6_col_{round_num}-93.npy", np.array(total_rewards))
        # print(np.array(total_rewards))
    #


# #painting
#     for i, agent_name in enumerate([ "agent_0", "agent_1"]):
#         plt.figure()
#         plt.plot(
#             np.arange(return_array.shape[0]) * 100,
#             rl_utils.moving_average(return_array[:, i], 9))
#         plt.xlabel("Episodes")
#         plt.ylabel("Returns")
#         plt.title(f"{agent_name} by QMADDPG")
#         plt.show()
#
#     total_rewards = np.sum(return_array, axis=1)

#     moving_avg_total_rewards = rl_utils.moving_average(total_rewards, 9)

#     plt.figure()
#     plt.plot(np.arange(len(moving_avg_total_rewards)), moving_avg_total_rewards)
#     plt.xlabel("Episodes")
#     plt.ylabel("Total Returns")
#     plt.title("Total Agent Rewards by QMADDPG")
#     plt.show()
#
#
#     state = env.reset()
#     flag = True
#     while flag :
#        
#         state = env.reset()
#         for i in range(1000):
#             actions = maddpg.take_action(env, state, explore=False)
#             next_state, reward, done, _ = env.step(actions)
#             # print(reward)
#             # print(env.agents[0].state.p_pos)
#             state = next_state
#             # print(state)
#             # print("State type:", type(state))
#             i = env.render()
#             if type(i)==int:
#                 flag = False
#                 break
#     time.sleep(0.05)# reward_list = np.array(reward_list)
#     # np.save(file="little/reward.npy", arr=reward_list)
#
#
#
#
#
#
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import time

    # state = env.reset()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    state = env.reset()

    while True:
        state = env.reset()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        x_data_agents = [[] for _ in range(len(env.agents))]  
        y_data_agents = [[] for _ in range(len(env.agents))]
        z_data_agents = [[] for _ in range(len(env.agents))]

        x_data_targets = [[] for _ in range(len(env.targets))]  
        y_data_targets = [[] for _ in range(len(env.targets))]
        z_data_targets = [[] for _ in range(len(env.targets))]

        for i in range(1000):
            actions = maddpg.take_action(env, state, explore=False)
            next_state, reward, done, _ = env.step(actions)
            state = next_state

            for agent_idx, agent in enumerate(env.agents):
                x_data_agents[agent_idx].append(agent.state.p_pos[0])
                y_data_agents[agent_idx].append(agent.state.p_pos[1])
                z_data_agents[agent_idx].append(agent.state.p_pos[2])

      
            for target_idx, target in enumerate(env.targets):
                x_data_targets[target_idx].append(target.state.p_pos[0])
                y_data_targets[target_idx].append(target.state.p_pos[1])
                z_data_targets[target_idx].append(target.state.p_pos[2])

        ax.clear()
     
        for agent_idx in range(len(env.agents)):
            agent_data = np.array(list(zip(x_data_agents[agent_idx],
                                           y_data_agents[agent_idx],
                                           z_data_agents[agent_idx])))
            np.savetxt(f"agent_{agent_idx}_trajectory.csv", agent_data, delimiter=",")

 
        for target_idx in range(len(env.targets)):
            target_data = np.array(list(zip(x_data_targets[target_idx],
                                            y_data_targets[target_idx],
                                            z_data_targets[target_idx])))
            np.savetxt(f"target_{target_idx}_trajectory.csv", target_data, delimiter=",")

        # print(x_data_agents)
    
        colors_agents = ['blue', 'green', 'orange', 'orange', 'blue', 'green', 'orange', 'orange', 'blue', 'green', 'orange', 'orange'] 
        for agent_idx in range(len(env.agents)):
            ax.plot(x_data_agents[agent_idx], y_data_agents[agent_idx], z_data_agents[agent_idx],
                    color=colors_agents[agent_idx], label=f'Agent {agent_idx + 1}')

        colors_targets = ['red', 'red', 'red', 'red']  
        for target_idx in range(len(env.targets)):
            ax.plot(x_data_targets[target_idx], y_data_targets[target_idx], z_data_targets[target_idx],
                    color=colors_targets[target_idx], label=f'Target {target_idx + 1}')

        plt.draw()

        plt.pause(5000)

    time.sleep(0.05)

