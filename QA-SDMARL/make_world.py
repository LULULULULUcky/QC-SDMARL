from environment import Agent, World, Entity, Target
import numpy as np
import math
import sympy
from multiagent.scenario import BaseScenario
from sklearn.cluster import KMeans
import random



class MakeWorld(BaseScenario):
    def make_world(self):
        world = World()

        num_agents = 6

        num_targets = 2
        world.num_agents = num_agents
        world.num_targets = num_targets

        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.size = 0.0110
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.movable = True
            # agent.em = np.random.uniform(0, 100, 2)

        world.targets = [Target() for i in range(num_targets)]
        for i, target in enumerate(world.targets):
            target.name = "target %d" % i
            target.size = 0.0110
            target.color = np.array([0.85, 0.35, 0.35])
            target.movable = True
        self.rest_world(world)
        return world

    def fuzzy_membership(self, value, range_min, range_max, inverse=False):
        if value < range_min:
            return 0 if not inverse else 1
        elif value > range_max:
            return 1 if not inverse else 0
        else:
            return (value - range_min) / (range_max - range_min)

    def fuzzy_set(self, value, set_type):
        if set_type == 'very_slow':
            return max(0.0, 1.0 - value / 40.0)
        elif set_type == 'slow':
            return max(0.0, min((value - 40.0) / 30.0, 1.0 - (value - 70.0) / 30.0))
        elif set_type == 'fast':
            return max(0.0, min((value - 70.0) / 20.0, 1.0 - (value - 90.0) / 10.0))
        elif set_type == 'very_fast':
            return max(0.0, (value - 90.0) / 10.0)
 
        return 0.0

    def complex_rule_based_scoring(self, row):
        score = 0
  
        avg_speed_score = sum(row['Speed_Scores'].values()) / len(row['Speed_Scores'])
        if avg_speed_score > 0.5:
            high_order_polynomial = avg_speed_score ** 3 + 2 * row['Acceleration_Score'] ** 2
            score += high_order_polynomial / (1 + np.exp(-row['Remaining_Battery_Score'])) * 4

        hyperbolic_cosine_term = np.cosh(row['Energy_Consumption_Score'] - row['Carrying_Capacity_Score'])
        radical_term = np.sqrt(np.abs(row['Carrying_Capacity_Score'] - row['Energy_Consumption_Score']))
        score += hyperbolic_cosine_term * radical_term * 2


        sensory_distance_interaction = np.exp(
            -np.abs(np.log(row['Sensory_Range_Score'] + 1) - np.log(row['Distance_to_Target1_Score'] + 1)))
        score += sensory_distance_interaction * 3


        sensory_distance_interaction_2 = np.exp(
            -np.abs(np.log(row['Sensory_Range_Score'] + 1) - np.log(row['Distance_to_Target2_Score'] + 1)))
        score += sensory_distance_interaction_2 * 3

        weighted_avg_score = (avg_speed_score ** 2 + row['Acceleration_Score'] ** 2 + row[
            'Energy_Consumption_Score'] ** 2) / 3
        score += weighted_avg_score * 2


        combined_exponential = np.exp(row['Acceleration_Score'] - avg_speed_score)
        inverse_function_score = 1 / (1 + combined_exponential)
        score += inverse_function_score * 3


        if row['Speed_Scores']['very_slow'] > 0.7 and row['Carrying_Capacity_Score'] > 0.7:
            score += row['Speed_Scores']['very_slow'] * row['Carrying_Capacity_Score'] * 5

     
        if row['Speed_Scores']['very_fast'] > 0.7 and row['Remaining_Battery_Score'] > 0.7:
            score += row['Speed_Scores']['very_fast'] * row['Remaining_Battery_Score'] * 4

        
        if (row['Speed_Scores']['slow'] > 0.5 or row['Speed_Scores']['fast'] > 0.5) and row[
            'Sensory_Range_Score'] > 0.7:
            score += max(row['Speed_Scores']['slow'], row['Speed_Scores']['fast']) * row['Sensory_Range_Score'] * 3

   
        return score

    def calculate_agent_scores(self, world):
        for agent in world.agents:
        
            scalar_speed = np.linalg.norm(agent.state.p_vel)

          
            speed_scores = {
                'very_slow': self.fuzzy_set(scalar_speed, 'very_slow'),
                'slow': self.fuzzy_set(scalar_speed, 'slow'),
                'fast': self.fuzzy_set(scalar_speed, 'fast'),
                'very_fast': self.fuzzy_set(scalar_speed, 'very_fast')
            }

         
            acceleration_score = self.fuzzy_membership(agent.acceleration, 20, 50)
            energy_consumption_score = 1 - self.fuzzy_membership(agent.energy_consumption_rate, 200, 1000)
            carrying_capacity_score = self.fuzzy_membership(agent.carrying_capacity, 10, 50)
            sensory_range_score = self.fuzzy_membership(agent.sensory_range, 50, 100)
            remaining_battery_score = self.fuzzy_membership(agent.bemaining_battery, 20, 100)

           
            distance_to_targets_scores = {}
            for i, target in enumerate(world.targets):
                distance = np.linalg.norm(agent.state.p_pos - target.state.p_pos)
                distance_to_targets_scores[f'Distance_to_Target{i + 1}_Score'] = self.fuzzy_membership(distance, 10, 50,
                                                                                                       True)

             
                agent_attributes = {
                    'Speed_Scores': speed_scores,
                    'Acceleration_Score': acceleration_score,
                    'Energy_Consumption_Score': energy_consumption_score,
                    'Carrying_Capacity_Score': carrying_capacity_score,
                    'Sensory_Range_Score': sensory_range_score,
                    'Remaining_Battery_Score': remaining_battery_score,
                    **distance_to_targets_scores
                }

        
            agent.score = self.complex_rule_based_scoring(agent_attributes)


    def rest_world(self, world):
        for i in range(0, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        world.agents[0].color = np.array([255, 0, 0])
        for i in range(0, world.num_targets):
            world.targets[i].color = np.array([0.85, 0.35, 0.35])
        for target in world.targets:
            target.state.p_pos = np.random.uniform(-0.5, -0.4, 3)
       
            random_perturbation = np.random.uniform(-0.02, 0.02, 3)
            target.state.p_vel = np.array([0,0,0]) + random_perturbation
            target.state.p_w = np.zeros(1)

        for agent in world.agents:  
      
            agent.state.p_pos = np.random.uniform(-0.6, -0.5, 3)
            # agent.state.p_vel = np.zeros(3)
            # agent.state.p_w = np.zeros(1)
            # agent.state.p_pos = np.random.uniform(-0.5, -0.7, 3)
            agent.state.p_vel = np.zeros(3)
            agent.state.p_w = np.zeros(1)

            agent.acceleration = np.random.uniform(0, 5) 
            agent.energy_consumption_rate = np.random.uniform(0.1, 2)  
            agent.carrying_capacity = np.random.randint(1, 10)  
            agent.sensory_range = np.random.uniform(10, 50) 
            agent.bemaining_battery = np.random.uniform(20, 100)  
            agent.energy = 100000

        self.calculate_agent_scores(world)

    
        sorted_agents = sorted(world.agents, key=lambda x: x.score, reverse=True)
   
        num_agents = len(sorted_agents)
        group_size = num_agents // 2
    
        groups = [sorted_agents[i:i + group_size] for i in range(0, num_agents, group_size)]

      
        # for agent in group1:
        #     agent.goal = world.targets[0]
        # for agent in group2:
        #     agent.goal = world.targets[1]

     
        for i, group in enumerate(groups):
            target = world.targets[i % len(world.targets)]  
            for agent in group:
                agent.goal = target




    def calculate_cluster_centers(self, world, num_clusters):

        if len(world.targets) == 0:
            return []

    
        target_positions = np.array([t.state.p_pos for t in world.targets])
        center = np.mean(target_positions, axis=0)

     
        new_target = Target()
        new_target.state.p_pos = center

        return [new_target]

    def get_agents(self, world):
        return [agent for agent in range(world.agents)]

    def get_target(self, world):
        return [target for target in range(world.targets)]

    def agent_reward(self, agent, world):
        pass
   
    def get_distance(self,entity_pos):

        abs_entity_pos = list(map(abs,entity_pos))

        fai_z = math.asin(abs_entity_pos[2] / math.sqrt(abs_entity_pos[0]**2+abs_entity_pos[1]**2+abs_entity_pos[2]**2))
        fai_z_fai = math.atan(abs_entity_pos[1]/abs_entity_pos[0])
 
        EM = -55.088*math.sqrt((abs_entity_pos[0]**2 + abs_entity_pos[1]**2+abs_entity_pos[2]**2))
        N = 30
      
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

    def observation(self, agent, world):

  

        # entity_pos = []

        # for entity in world.targets:
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        #
        # print(entity_pos[0])

   
        # entity_pos.append(self.calculate_cluster_centers(world, num_clusters=1)[0].state.p_pos - agent.state.p_pos)

        if agent.goal is not None:
            entity_pos = [agent.goal.state.p_pos - agent.state.p_pos]
            entity_pos = [self.get_distance(entity_pos[0])]
        else:
            entity_pos = [0,0,0]

        entity_pos = [self.get_distance(entity_pos[0])]
        other_pos = []
 
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

        aagent_velocity_data = 0
        wei_rew = 0
        entity_pos = self.observation(agent, world)  
        d_target_min = 0.05 
        d_auv_min = 0.015 
        w = 2
        d = [math.sqrt(entity_pos[0] ** 2 + entity_pos[1] ** 2 + entity_pos[2] ** 2)]  
        de = []

        # 计算智能体与目标的坐标差异
        for agent in world.agents:
            de.append(abs(agent.state.p_pos[0] - agent.goal.state.p_pos[0]))  
            de.append(abs(agent.state.p_pos[1] - agent.goal.state.p_pos[1]))  
            de.append(abs(agent.state.p_pos[2] - agent.goal.state.p_pos[2])) 

        distance_error1 = 0
        for i in range(3, 18, 3):
            if entity_pos[i] == 0 and entity_pos[i + 1] == 0 and entity_pos[i + 2] == 0:
                continue
            distance_error1 += abs(
                math.sqrt(entity_pos[i] ** 2 + entity_pos[i + 1] ** 2 + entity_pos[i + 2] ** 2) - d_auv_min)
        d.append(distance_error1)

 
        distance_error = abs(d[0] - d_target_min)
        pos_rew = -w * distance_error  
        col_rew = -w * d[1]

   
        velocity_change = np.linalg.norm(agent.state.p_vel - agent.previous_velocity)
        velocity_change_threshold = 0.01
        velocity_penalty = max(0, velocity_change - velocity_change_threshold)
        ocean_reward = -1.0 * (velocity_penalty) 

     
        agent.previous_velocity = np.copy(agent.state.p_vel)

        target_direction = np.array([entity_pos[0], entity_pos[1], entity_pos[2]])
        agent_direction = agent.state.p_vel
        cos_theta = np.dot(target_direction, agent_direction) / (
                np.linalg.norm(target_direction) * np.linalg.norm(agent_direction) + 1e-6)
        direction_reward = cos_theta  


        agent.previous_velocity = np.copy(agent.state.p_vel)
        for i, agent in enumerate(world.agents):

            agent_velocity_magnitude = np.linalg.norm(agent.state.p_vel)
            target_velocity_magnitude = np.linalg.norm(agent.goal.state.p_vel)
            velocity_difference = abs(agent_velocity_magnitude - target_velocity_magnitude)

            if velocity_difference > 0.0:
  
                aagent_velocity_data -= velocity_difference
            else:
           
                aagent_velocity_data -= 0  # 或者替换为一个较小的值，比如 agent_velocity_data[i] -= velocity_difference * 0.1

  
        distribution_reward = 0
        num_agents = len(world.agents)
        for i, other_agent in enumerate(world.agents):
            if agent != other_agent:
            
                distance_to_other = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
         
                if distance_to_other < 0.01: 
                    distribution_reward -= 1 / (distance_to_other + 1e-6) 

        proximity_penalty = 0
        for i, other_agent in enumerate(world.agents):
            if other_agent is not agent:
                dist = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
           
                if dist < 0.1:  
                    proximity_penalty += (0.1 - dist) * 10  
        return 0.4 * distribution_reward + 0.6 * pos_rew + 0.4 * col_rew + 0.01 * ocean_reward + 0.1 * aagent_velocity_data + 1 * direction_reward - 0.2 * proximity_penalty
