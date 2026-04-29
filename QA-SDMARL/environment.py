
import random

import numpy as np




class Entity(object):
    def __init__(self):
        self.name = ''

        self.size = 0.001

        self.movable = False
  
        self.collide = True
        self.color = None

        self.max_speed = 0.031
        self.max_angular_speed = None
        self.state = EntityState()


class EntityState(object):
    def __init__(self):

        self.p_pos = None

        self.p_vel = None

        self.p_w = None





class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()

        self.movable = True
        self.silent = False
  
        self.u_range = 10.0
  
        self.state = AgentState()
 
        self.action = Action()

        self.em = None
        self.energy = 100000
        self.u_noise = None
        self.initial_mass = 1.0
        self.best = False
        self.goal = None 
        self.acceleration = 0
        self.energy_consumption_rate = 0
        self.carrying_capacity = 0
        self.sensory_range = 0
        self.bemaining_battery = 0
        self.lift_coefficient = 1.0

        self.drag_coefficient = 0.5  

        self.area = 0.1 

        self.volume = 0.2  
        self.previous_velocity = 0

    def set_goal(self, target):

        if isinstance(target, Target):
            self.goal = target
        else:
            raise ValueError("Goal must be a Target instance")

    @property
    def mass(self):
        return self.initial_mass





class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()

        self.p_com = False





class Action(object):
    def __init__(self):

        self.act_u = None

        self.act_c = None




class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()




class Target(Entity):
    def __init__(self):
        super(Target, self).__init__()
        self.movable = True

        self.state = TargetState()


class TargetState(EntityState):
    def __init__(self):
        super(TargetState, self).__init__()
        self.p_vel = None

class World(object):
    def __init__(self):
  
        self.agents = []
        self.landmarks = []
        self.targets = []

        self.dim_p = 3

        self.dt = 0.1

        self.damping = 0.25

        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        self.fluid_properties = {
            'density': 1000, 
            'viscosity': 1e-3 
        }


    @property
    def entities(self):
        return self.agents + self.landmarks + self.targets


    def step(self):

        p_force = [None] * len(self.agents)

        p_force = self.apply_action_force(p_force)

        p_force = self.apply_environment_force(p_force)

        self.integrate_state(p_force)
        # update agent state
        # for agent in self.agents:
        #     agent.em = np.random.uniform(0,100,2)

    def simulate_navier_stokes(self, agent):
       

        noise = self.generate_noise(agent)
        fluid_velocity = np.array([1.0, 1.0, 1.0]) + noise
        return fluid_velocity

    def generate_noise(self, agent):
        

        noise = np.random.normal(0, 0.1, 3)
        return noise

    def calculate_drag_force(self, agent, fluid_velocity):
       
        drag_magnitude = 0.5 * self.fluid_properties['density'] * np.linalg.norm(
            agent.state.p_vel) ** 2 * agent.drag_coefficient * agent.area
        drag_direction = -agent.state.p_vel / np.linalg.norm(agent.state.p_vel) if np.linalg.norm(
            agent.state.p_vel) > 0 else np.zeros(3)
        drag_force = drag_magnitude * drag_direction
        return drag_force

    def calculate_lift_force(self, agent, fluid_velocity):
      
        lift_magnitude = 0.5 * self.fluid_properties['density'] * np.linalg.norm(
            fluid_velocity) ** 2 * agent.lift_coefficient * agent.area
        lift_force = lift_magnitude * np.array([0, -1, 0])  
        return lift_force

    def calculate_morison_force(self, agent, fluid_velocity):
        
        wave_velocity_change = 0.5
        morison_magnitude = self.fluid_properties['density'] * agent.volume * wave_velocity_change
  
        relative_velocity = fluid_velocity - agent.state.p_vel
        morison_direction = relative_velocity / np.linalg.norm(relative_velocity) if np.linalg.norm(
            relative_velocity) > 0 else np.zeros(3)
        morison_force = morison_magnitude * morison_direction
        return morison_force

    def simulate_navier_stokes(self, agent):
        
        k = 1.0  
        epsilon = 1.0 
        nu_t = k ** 2 / epsilon  
        u_prime = np.random.normal(0, np.sqrt(nu_t), 3)  

  
        u_bar = np.array([1.0, 1.0, 1.0])

     
        fluid_velocity = u_bar + u_prime
        return fluid_velocity

    def apply_action_force(self, p_force):
        for i, agent in enumerate(self.agents):
            if agent.movable:
          
                fluid_velocity = self.simulate_navier_stokes(agent)

         
                lift_force = self.calculate_lift_force(agent, fluid_velocity)
                drag_force = self.calculate_drag_force(agent, fluid_velocity)
                morison_force = self.calculate_morison_force(agent, fluid_velocity)

          
                total_force = lift_force / 40000 + drag_force / 10000 + morison_force / 3000 + agent.action.u

   
                p_force[i] = total_force
        return p_force


    def apply_environment_force(self, p_force):

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

            target_speed = 0.03

 
            if target.state.p_vel is None or np.linalg.norm(target.state.p_vel) == 0:
                random_direction = np.random.normal(0, 1, 3)
                random_direction /= np.linalg.norm(random_direction) 
                target.state.p_vel = random_direction * target_speed


            random_perturbation = np.random.uniform(-0.0015, 0.0015, 3)


            new_direction = target.state.p_vel + random_perturbation
            new_direction /= np.linalg.norm(new_direction)  
            target.state.p_vel = new_direction * target_speed


            target.state.p_pos += target.state.p_vel * self.dt

 
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
     
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
