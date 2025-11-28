from environment import Agent, World, Entity, Target
import numpy as np
import math
import sympy
from multiagent.scenario import BaseScenario
from sklearn.cluster import KMeans
import random
"""创造实体存在的世界"""


class MakeWorld(BaseScenario):
    def make_world(self):
        world = World()
        # 智能体的数量
        num_agents = 6
        # 目标的数量
        num_targets = 2
        world.num_agents = num_agents
        world.num_targets = num_targets
        # 添加智能体
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.size = 0.0110
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.movable = True
            # agent.em = np.random.uniform(0, 100, 2)
        # 添加目标
        world.targets = [Target() for i in range(num_targets)]
        for i, target in enumerate(world.targets):
            target.name = "target %d" % i
            target.size = 0.0110
            target.color = np.array([0.85, 0.35, 0.35])
            target.movable = True
        self.rest_world(world)
        return world

    # 定义模糊逻辑评分函数
    # 模糊逻辑评分函数（修改为更通用的形式）
    def fuzzy_membership(self, value, range_min, range_max, inverse=False):
        if value < range_min:
            return 0 if not inverse else 1
        elif value > range_max:
            return 1 if not inverse else 0
        else:
            return (value - range_min) / (range_max - range_min)

    # 模糊集合定义函数
    def fuzzy_set(self, value, set_type):
        if set_type == 'very_slow':
            return max(0.0, 1.0 - value / 40.0)
        elif set_type == 'slow':
            return max(0.0, min((value - 40.0) / 30.0, 1.0 - (value - 70.0) / 30.0))
        elif set_type == 'fast':
            return max(0.0, min((value - 70.0) / 20.0, 1.0 - (value - 90.0) / 10.0))
        elif set_type == 'very_fast':
            return max(0.0, (value - 90.0) / 10.0)
        # ... [为其他属性添加更多模糊集合]
        return 0.0

    def complex_rule_based_scoring(self, row):
        score = 0
        # 能量调节的性能加成
        avg_speed_score = sum(row['Speed_Scores'].values()) / len(row['Speed_Scores'])
        if avg_speed_score > 0.5:
            high_order_polynomial = avg_speed_score ** 3 + 2 * row['Acceleration_Score'] ** 2
            score += high_order_polynomial / (1 + np.exp(-row['Remaining_Battery_Score'])) * 4

        # 能效载重平衡评估
        hyperbolic_cosine_term = np.cosh(row['Energy_Consumption_Score'] - row['Carrying_Capacity_Score'])
        radical_term = np.sqrt(np.abs(row['Carrying_Capacity_Score'] - row['Energy_Consumption_Score']))
        score += hyperbolic_cosine_term * radical_term * 2

        # 感知与目标距离一致性
        sensory_distance_interaction = np.exp(
            -np.abs(np.log(row['Sensory_Range_Score'] + 1) - np.log(row['Distance_to_Target1_Score'] + 1)))
        score += sensory_distance_interaction * 3

        # 双目标感知一致性评估
        sensory_distance_interaction_2 = np.exp(
            -np.abs(np.log(row['Sensory_Range_Score'] + 1) - np.log(row['Distance_to_Target2_Score'] + 1)))
        score += sensory_distance_interaction_2 * 3

        # 综合性能指标加成
        weighted_avg_score = (avg_speed_score ** 2 + row['Acceleration_Score'] ** 2 + row[
            'Energy_Consumption_Score'] ** 2) / 3
        score += weighted_avg_score * 2

        # 动态逆反应性能评估
        combined_exponential = np.exp(row['Acceleration_Score'] - avg_speed_score)
        inverse_function_score = 1 / (1 + combined_exponential)
        score += inverse_function_score * 3

        # 精细操作专长加分
        if row['Speed_Scores']['very_slow'] > 0.7 and row['Carrying_Capacity_Score'] > 0.7:
            score += row['Speed_Scores']['very_slow'] * row['Carrying_Capacity_Score'] * 5

        # 远程快速响应优势
        if row['Speed_Scores']['very_fast'] > 0.7 and row['Remaining_Battery_Score'] > 0.7:
            score += row['Speed_Scores']['very_fast'] * row['Remaining_Battery_Score'] * 4

        # 避障能力评估
        if (row['Speed_Scores']['slow'] > 0.5 or row['Speed_Scores']['fast'] > 0.5) and row[
            'Sensory_Range_Score'] > 0.7:
            score += max(row['Speed_Scores']['slow'], row['Speed_Scores']['fast']) * row['Sensory_Range_Score'] * 3

        # ... [继续添加更多复杂的规则]
        return score

    def calculate_agent_scores(self, world):
        for agent in world.agents:
            # 将速度向量转换为标量
            scalar_speed = np.linalg.norm(agent.state.p_vel)

            # 计算模糊逻辑评分的隶属度
            speed_scores = {
                'very_slow': self.fuzzy_set(scalar_speed, 'very_slow'),
                'slow': self.fuzzy_set(scalar_speed, 'slow'),
                'fast': self.fuzzy_set(scalar_speed, 'fast'),
                'very_fast': self.fuzzy_set(scalar_speed, 'very_fast')
            }

            # 使用fuzzy_membership函数计算其他属性的隶属度
            acceleration_score = self.fuzzy_membership(agent.acceleration, 20, 50)
            energy_consumption_score = 1 - self.fuzzy_membership(agent.energy_consumption_rate, 200, 1000)
            carrying_capacity_score = self.fuzzy_membership(agent.carrying_capacity, 10, 50)
            sensory_range_score = self.fuzzy_membership(agent.sensory_range, 50, 100)
            remaining_battery_score = self.fuzzy_membership(agent.bemaining_battery, 20, 100)

            # 计算每个智能体与每个目标之间的距离分数
            distance_to_targets_scores = {}
            for i, target in enumerate(world.targets):
                distance = np.linalg.norm(agent.state.p_pos - target.state.p_pos)
                distance_to_targets_scores[f'Distance_to_Target{i + 1}_Score'] = self.fuzzy_membership(distance, 10, 50,
                                                                                                       True)

                # 将隶属度添加到属性字典中
                agent_attributes = {
                    'Speed_Scores': speed_scores,
                    'Acceleration_Score': acceleration_score,
                    'Energy_Consumption_Score': energy_consumption_score,
                    'Carrying_Capacity_Score': carrying_capacity_score,
                    'Sensory_Range_Score': sensory_range_score,
                    'Remaining_Battery_Score': remaining_battery_score,
                    **distance_to_targets_scores
                }

            # 计算总评分
            agent.score = self.complex_rule_based_scoring(agent_attributes)

    # 重置世界中智能体和目标的位置与速度，应该是在仿真开始的时候进行初始化的，这里面涉及到AUV目标（goal）的设定，在这里改一下应该就行
    def rest_world(self, world):
        for i in range(0, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        world.agents[0].color = np.array([255, 0, 0])
        for i in range(0, world.num_targets):
            world.targets[i].color = np.array([0.85, 0.35, 0.35])
        for target in world.targets:
            target.state.p_pos = np.random.uniform(-0.5, -0.4, 3)
            # 为速度的每个分量添加随机扰动
            random_perturbation = np.random.uniform(-0.02, 0.02, 3)
            target.state.p_vel = np.array([0,0,0]) + random_perturbation
            target.state.p_w = np.zeros(1)

        for agent in world.agents:  # 更改
            # agent.goal = self.calculate_cluster_centers(world, num_clusters=1)  # 假设使用1个聚类中心
            agent.state.p_pos = np.random.uniform(-0.6, -0.5, 3)
            # agent.state.p_vel = np.zeros(3)
            # agent.state.p_w = np.zeros(1)
            # agent.state.p_pos = np.random.uniform(-0.5, -0.7, 3)
            agent.state.p_vel = np.zeros(3)
            agent.state.p_w = np.zeros(1)

            agent.acceleration = np.random.uniform(0, 5)  # 假设加速度在0到5之间
            agent.energy_consumption_rate = np.random.uniform(0.1, 2)  # 能耗率在0.1到2之间
            agent.carrying_capacity = np.random.randint(1, 10)  # 承载能力为1到10之间的整数
            agent.sensory_range = np.random.uniform(10, 50)  # 感应范围在10到50之间
            agent.bemaining_battery = np.random.uniform(20, 100)  # 剩余电池电量在20%到100%之间
            agent.energy = 100000

        self.calculate_agent_scores(world)

        # 将智能体按分数排序并分组
        sorted_agents = sorted(world.agents, key=lambda x: x.score, reverse=True)
        # 计算每个组的大小（尽可能均匀分配）
        num_agents = len(sorted_agents)
        group_size = num_agents // 2#更改
        # 分配智能体到四个组
        groups = [sorted_agents[i:i + group_size] for i in range(0, num_agents, group_size)]

        # # 为每个智能体设置目标
        # for agent in group1:
        #     agent.goal = world.targets[0]
        # for agent in group2:
        #     agent.goal = world.targets[1]

        # 为每个组内的智能体设置目标
        for i, group in enumerate(groups):
            target = world.targets[i % len(world.targets)]  # 确保目标索引不会超出范围
            for agent in group:
                agent.goal = target



    # wsb:新加的
    def calculate_cluster_centers(self, world, num_clusters):
        # 检查是否有目标，如果没有则返回空列表
        if len(world.targets) == 0:
            return []

        # 计算目标位置的平均值
        target_positions = np.array([t.state.p_pos for t in world.targets])
        center = np.mean(target_positions, axis=0)

        # 创建一个新的目标对象，其位置为计算出的平均位置
        new_target = Target()
        new_target.state.p_pos = center

        return [new_target]

    def get_agents(self, world):
        return [agent for agent in range(world.agents)]

    def get_target(self, world):
        return [target for target in range(world.targets)]

    def agent_reward(self, agent, world):
        pass
    #计算距离，计算夹角，但是这里直接用的是坐标来计算角度然后应用声纳来计算距离，声纳部分是不是有点多此一举啦？
    def get_distance(self,entity_pos):
        #得到夹角信息
        abs_entity_pos = list(map(abs,entity_pos))
        # 计算角
        fai_z = math.asin(abs_entity_pos[2] / math.sqrt(abs_entity_pos[0]**2+abs_entity_pos[1]**2+abs_entity_pos[2]**2))
        fai_z_fai = math.atan(abs_entity_pos[1]/abs_entity_pos[0])
        # 计算回波强度
        EM = -55.088*math.sqrt((abs_entity_pos[0]**2 + abs_entity_pos[1]**2+abs_entity_pos[2]**2))
        N = 30
        # 计算距离d
        SL = 200
        TS = 3
        NL = 30
        DI = math.log10(N)
        DT = 0
        TL = (SL - EM + TS - (NL - DI) - DT) / 2
        f = 10 ** 4
        alf = 0.11 * (f * f) / (1 + f * f) + 44 * (f * f) / (4100 + f * f) + 2.75 * 10 ** (-4) * f * f + 0.003
        d = 10 ** 3 * (TL + 20) / alf
        dx = (d-3.8934) * math.cos(fai_z) * math.cos(fai_z_fai)
        dy = (d-3.8934) * math.cos(fai_z) * math.sin(fai_z_fai)
        dz = (d-3.8934) * math.sin(fai_z)

        if entity_pos[0] / dx < 0:
            dx = -1 * dx
        if entity_pos[1] / dy < 0:
            dy = -1 * dy
        if entity_pos[2] / dz < 0:
            dz = -1 * dz
        d1 = [dx,dy,dz]
        return np.array(d1)
    # 通过实际的传感器强度获取距离信息，获取当前观察者和其他实体的距离:
    def observation(self, agent, world):

        # 记录他们之间的位置关系

        # entity_pos = []
        # 计算观察者和目标之间的距离
        # for entity in world.targets:
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        #
        # print(entity_pos[0])

        #wsb：这里更改上面那个循环，因为压根没用到之前设置的goal当作目标点，那个goal属性完全是摆设
        # entity_pos.append(self.calculate_cluster_centers(world, num_clusters=1)[0].state.p_pos - agent.state.p_pos)

        if agent.goal is not None:
            entity_pos = [agent.goal.state.p_pos - agent.state.p_pos]
            entity_pos = [self.get_distance(entity_pos[0])]
        else:
            entity_pos = [0,0,0]

        entity_pos = [self.get_distance(entity_pos[0])]
        other_pos = []
        # 计算观察者和其他智能体之间的距离
        for other in world.agents:
            if other is agent:
                continue
            if other.goal is not None and other.goal == agent.goal:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
            else:
                other_pos.append([0, 0, 0])
        for i in range(len(other_pos)):
            if other_pos[i][0] == 0 and other_pos[i][1] == 0 and other_pos[i][2] == 0:
                continue
            other_pos[i] = self.get_distance(other_pos[i])
        return np.concatenate(entity_pos + other_pos)

    def reward(self, agent, world):
        # 初始化奖励和位置数据
        aagent_velocity_data = 0
        wei_rew = 0
        entity_pos = self.observation(agent, world)  # 获取智能体的观察位置
        d_target_min = 0.05  # 目标的最小距离
        d_auv_min = 0.015  # AUV最小距离
        w = 2
        d = [math.sqrt(entity_pos[0] ** 2 + entity_pos[1] ** 2 + entity_pos[2] ** 2)]  # 计算与目标的距离
        de = []

        # 计算智能体与目标的坐标差异
        for agent in world.agents:
            de.append(abs(agent.state.p_pos[0] - agent.goal.state.p_pos[0]))  # x轴偏移
            de.append(abs(agent.state.p_pos[1] - agent.goal.state.p_pos[1]))  # y轴偏移
            de.append(abs(agent.state.p_pos[2] - agent.goal.state.p_pos[2]))  # z轴偏移

        distance_error1 = 0
        for i in range(3, 18, 3):
            if entity_pos[i] == 0 and entity_pos[i + 1] == 0 and entity_pos[i + 2] == 0:
                continue
            distance_error1 += abs(
                math.sqrt(entity_pos[i] ** 2 + entity_pos[i + 1] ** 2 + entity_pos[i + 2] ** 2) - d_auv_min)
        d.append(distance_error1)

        # 目标距离的误差和奖励
        distance_error = abs(d[0] - d_target_min)
        pos_rew = -w * distance_error  # 如果距离小于目标值，则给予负奖励（减少距离）
        col_rew = -w * d[1]

        # 计算速度变化（用来惩罚过大的速度变化）
        velocity_change = np.linalg.norm(agent.state.p_vel - agent.previous_velocity)
        velocity_change_threshold = 0.01
        velocity_penalty = max(0, velocity_change - velocity_change_threshold)
        ocean_reward = -1.0 * (velocity_penalty)  # 过大的速度变化给予惩罚

        # 更新智能体的上一时间步速度
        agent.previous_velocity = np.copy(agent.state.p_vel)

        # 计算运动方向与目标方向的夹角
        target_direction = np.array([entity_pos[0], entity_pos[1], entity_pos[2]])
        agent_direction = agent.state.p_vel
        cos_theta = np.dot(target_direction, agent_direction) / (
                np.linalg.norm(target_direction) * np.linalg.norm(agent_direction) + 1e-6)
        direction_reward = cos_theta  # 根据方向的余弦相似度给予奖励

        # 更新智能体的上一时间步速度
        agent.previous_velocity = np.copy(agent.state.p_vel)
        for i, agent in enumerate(world.agents):
            # 计算智能体速度的长度
            agent_velocity_magnitude = np.linalg.norm(agent.state.p_vel)
            target_velocity_magnitude = np.linalg.norm(agent.goal.state.p_vel)
            velocity_difference = abs(agent_velocity_magnitude - target_velocity_magnitude)

            # 检查速度差是否超过阈值，并应用惩罚
            if velocity_difference > 0.0:
                # 如果超过阈值，记录这个差值
                aagent_velocity_data -= velocity_difference
            else:
                # 如果没有超过阈值，可以选择不做操作，或者记录一个较小的值或零
                aagent_velocity_data -= 0  # 或者替换为一个较小的值，比如 agent_velocity_data[i] -= velocity_difference * 0.1

        # 计算分散性奖励
        # 我们希望智能体分布在目标的不同区域，避免聚集在同一位置。
        distribution_reward = 0
        num_agents = len(world.agents)
        for i, other_agent in enumerate(world.agents):
            if agent != other_agent:
                # 计算智能体之间的相对距离，越远表示它们分布得更分散
                distance_to_other = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
                # 通过惩罚过于接近的智能体来鼓励它们分散
                if distance_to_other < 0.01:  # 如果距离小于阈值，表示它们过于接近
                    distribution_reward -= 1 / (distance_to_other + 1e-6)  # 给予惩罚（距离越近惩罚越大）

        # 计算智能体之间的距离差异
        proximity_penalty = 0
        for i, other_agent in enumerate(world.agents):
            if other_agent is not agent:
                dist = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
                # 惩罚过于接近的AUV
                if dist < 0.1:  # 可以调整这个阈值
                    proximity_penalty += (0.1 - dist) * 10  # 惩罚因接近而增加的差距
        return 0.4 * distribution_reward + 0.6 * pos_rew + 0.4 * col_rew + 0.01 * ocean_reward + 0.1 * aagent_velocity_data + 1 * direction_reward - 0.2 * proximity_penalty