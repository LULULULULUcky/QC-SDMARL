"""实现强化学习编程中的环境，实体"""
import random

import numpy as np

"""实现实体类，包括移动和不可移动实体"""


class Entity(object):
    def __init__(self):
        self.name = ''
        # 实体大小
        self.size = 0.001
        # 实体是否可移动
        self.movable = False
        # 实体之间是否防止碰撞
        self.collide = True
        self.color = None
        # 设置实体间的速度和角速度
        self.max_speed = 0.031
        self.max_angular_speed = None
        self.state = EntityState()


"""实体的状态"""


class EntityState(object):
    def __init__(self):
        # 实体位置
        self.p_pos = None
        # 实体速度
        self.p_vel = None
        # 实体角速度
        self.p_w = None


"""定义可移动的智能体"""


class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # 可移动但是不能交流
        self.movable = True
        self.silent = False
        # 控制智能体的范围
        self.u_range = 10.0
        # 智能体的状态
        self.state = AgentState()
        # 智能体的动作
        self.action = Action()
        # 每个智能体的回波强度
        self.em = None
        self.energy = 100000
        self.u_noise = None
        self.initial_mass = 1.0
        self.best = False
        self.goal = None  # 添加goal属性，初始值为None
        self.acceleration = 0
        self.energy_consumption_rate = 0
        self.carrying_capacity = 0
        self.sensory_range = 0
        self.bemaining_battery = 0
        self.lift_coefficient = 1.0
        # 阻力系数
        self.drag_coefficient = 0.5  # 示例值，根据智能体的流体动力学设计调整
        # 迎风面积
        self.area = 0.1  # 示例值，根据智能体的实际面积来设置
        # 体积
        self.volume = 0.2  # 示例值，根据智能体的实际体积来设置
        self.previous_velocity = 0

    def set_goal(self, target):
        """设置Agent的目标"""
        if isinstance(target, Target):
            self.goal = target
        else:
            raise ValueError("Goal must be a Target instance")

    @property
    def mass(self):
        return self.initial_mass


"""定义可移动智能体的状态"""


class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # 智能体间是否可以通信
        self.p_com = False


"""定义智能体的动作"""


class Action(object):
    def __init__(self):
        # 智能体间的物理动作
        self.act_u = None
        # 智能体间的通信动作
        self.act_c = None


"""定义地标或障碍物"""


class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()


"""定义目标"""


class Target(Entity):
    def __init__(self):
        super(Target, self).__init__()
        self.movable = True
        #wsb：这里我在最后加了个（）
        self.state = TargetState()


class TargetState(EntityState):
    def __init__(self):
        super(TargetState, self).__init__()
        self.p_vel = None


"""定义多智能体世界"""


class World(object):
    def __init__(self):
        # 世界中的智能体和障碍物
        self.agents = []
        self.landmarks = []
        self.targets = []
        # 位置维度
        self.dim_p = 3
        # 仿真步长
        self.dt = 0.1
        # 阻尼
        self.damping = 0.25
        # 通信参数
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # 流体属性
        self.fluid_properties = {
            'density': 1000,  # 流体密度，单位 kg/m^3
            'viscosity': 1e-3  # 流体动力粘度，单位 Pa·s 或者 N·s/m^2
        }

    # 返回所有实体
    @property
    def entities(self):
        return self.agents + self.landmarks + self.targets

    # 根据动作更新世界，主要体现在位置更新
    def step(self):

        p_force = [None] * len(self.agents)
        # 考虑噪声
        p_force = self.apply_action_force(p_force)
        # 环境因素（我觉得更多是考虑碰撞）
        p_force = self.apply_environment_force(p_force)
        # 更新位置
        self.integrate_state(p_force)
        # update agent state
        # for agent in self.agents:
        #     agent.em = np.random.uniform(0,100,2)

    def simulate_navier_stokes(self, agent):
        """
        模拟纳维-斯托克斯方程以获得流体速度场。
        返回给定智能体的流体速度向量。
        """
        # 使用Perlin噪声或其他噪声函数来模拟流体速度的动态变化
        noise = self.generate_noise(agent)
        fluid_velocity = np.array([1.0, 1.0, 1.0]) + noise
        return fluid_velocity

    def generate_noise(self, agent):
        """
        生成噪声以模拟流体速度的变化。
        """
        # 噪声生成示例，可以根据需要使用更复杂的噪声函数
        noise = np.random.normal(0, 0.1, 3)
        return noise

    def calculate_drag_force(self, agent, fluid_velocity):
        """
        计算阻力，返回三维向量。
        """
        drag_magnitude = 0.5 * self.fluid_properties['density'] * np.linalg.norm(
            agent.state.p_vel) ** 2 * agent.drag_coefficient * agent.area
        drag_direction = -agent.state.p_vel / np.linalg.norm(agent.state.p_vel) if np.linalg.norm(
            agent.state.p_vel) > 0 else np.zeros(3)
        drag_force = drag_magnitude * drag_direction
        return drag_force

    def calculate_lift_force(self, agent, fluid_velocity):
        """
        计算升力，返回三维向量。
        """
        lift_magnitude = 0.5 * self.fluid_properties['density'] * np.linalg.norm(
            fluid_velocity) ** 2 * agent.lift_coefficient * agent.area
        lift_force = lift_magnitude * np.array([0, -1, 0])  # 假设升力垂直向下
        return lift_force

    def calculate_morison_force(self, agent, fluid_velocity):
        """
        计算Morison方程的作用力，返回三维向量。
        """
        wave_velocity_change = 0.5
        morison_magnitude = self.fluid_properties['density'] * agent.volume * wave_velocity_change
        # 假设Morison力与流体速度和智能体速度的差值方向相同
        relative_velocity = fluid_velocity - agent.state.p_vel
        morison_direction = relative_velocity / np.linalg.norm(relative_velocity) if np.linalg.norm(
            relative_velocity) > 0 else np.zeros(3)
        morison_force = morison_magnitude * morison_direction
        return morison_force

    def simulate_navier_stokes(self, agent):
        """
        模拟雷诺平均纳维-斯托克斯方程（RANS）以获得流体速度场。
        返回给定智能体的流体速度向量。
        """
        # 使用湍流模型来模拟流体速度的变化，简化版本
        # 这里使用k-epsilon模型作为示例
        k = 1.0  # 湍动能
        epsilon = 1.0  # 湍动能耗散率
        nu_t = k ** 2 / epsilon  # 湍流粘性系数
        u_prime = np.random.normal(0, np.sqrt(nu_t), 3)  # 湍流脉动速度

        # 平均流速假设为一定值
        u_bar = np.array([1.0, 1.0, 1.0])

        # 总流速 = 平均流速 + 脉动流速
        fluid_velocity = u_bar + u_prime
        return fluid_velocity

    def apply_action_force(self, p_force):
        for i, agent in enumerate(self.agents):
            if agent.movable:
                # 计算雷诺平均纳维-斯托克斯方程的流体速度
                fluid_velocity = self.simulate_navier_stokes(agent)

                # 计算升力、阻力和Morison方程的作用力
                lift_force = self.calculate_lift_force(agent, fluid_velocity)
                drag_force = self.calculate_drag_force(agent, fluid_velocity)
                morison_force = self.calculate_morison_force(agent, fluid_velocity)

                # 将所有力合并为总力
                total_force = lift_force / 40000 + drag_force / 10000 + morison_force / 3000 + agent.action.u

                # 把总力赋值给智能体的力数组
                p_force[i] = total_force
        return p_force

    # 离散动作的物理力，主要是进行了碰撞的监测和反馈，如果碰撞了就会把碰撞的力加到总的力上
    def apply_environment_force(self, p_force):
        # 碰撞
        for a, entity_a in enumerate(self.agents):
            for b, entity_b in enumerate(self.agents):
                if (b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if (f_a is not None):
                    if (p_force[a] is None): p_force[a] = 0.0

                    p_force[a] = f_a + p_force[a]
                if (f_b is not None):
                    if (p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # 更新状态，根据结算结果的力对位置和速度等变量进行更新，还进行了最大速度限制等操作
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.agents):
            if not entity.movable: continue
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])+ np.square(entity.state.p_vel[2]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(
                                                                          entity.state.p_vel[1])+np.square(entity.state.p_vel[2])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

        # for target in self.targets:
        #     target.state.p_pos[0] += (target.state.p_vel[0] + random.uniform(-0.02, 0.02)) * self.dt
        #     target.state.p_pos[1] += (target.state.p_vel[1] + random.uniform(-0.02, 0.02)) * self.dt
        #     target.state.p_pos[2] += (target.state.p_vel[2] + random.uniform(-0.02, 0.02)) * self.dt

        for target in self.targets:
            # 定义目标速度的大小
            target_speed = 0.03

            # 若目标当前没有速度，则初始化一个具有特定大小和随机方向的速度
            if target.state.p_vel is None or np.linalg.norm(target.state.p_vel) == 0:
                random_direction = np.random.normal(0, 1, 3)
                random_direction /= np.linalg.norm(random_direction)  # 归一化以获得单位向量
                target.state.p_vel = random_direction * target_speed

            # 生成一个小的随机偏移量，用于改变方向
            random_perturbation = np.random.uniform(-0.0015, 0.0015, 3)

            # 将随机偏移量添加到当前速度上，然后归一化并乘以速度大小以确保速度标量不变
            new_direction = target.state.p_vel + random_perturbation
            new_direction /= np.linalg.norm(new_direction)  # 归一化以获得单位向量
            target.state.p_vel = new_direction * target_speed

            # 根据更新后的速度计算新的位置
            target.state.p_pos += target.state.p_vel * self.dt

    # 考虑实体间的碰撞，具体的碰撞力计算函数
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        # 计算两个实体之间的距离
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]