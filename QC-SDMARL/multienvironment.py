import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
import math

all_entity_positions = [[] for _ in range(6)]  
agent_velocity_data = [[] for _ in range(6)]  
target_velocity_data = [[] for _ in range(2)]  
class Multiagent(gym.Env):
    def __init__(self, world,type, reset_callback=None, reward_callback=None, observation_callback=None, shared_viewer=True):
        super(Multiagent, self).__init__()
        self.world = world
        self.agents = world.agents
        self.type = type
        self.targets = world.targets
        self.n = len(world.agents)
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback

        self.action_space = []
        self.observation_space = []

        for agent in self.agents:
 
            # u_action_sapce = spaces.Box(low=-agent.u_range,high=+agent.u_range,shape = (2,),dtype=np.float)
            u_action_space = spaces.Discrete(2 * world.dim_p + 1)
            self.action_space.append(u_action_space)

      
            obs_dim = len(type.observation(agent, self.world))
            # obs_dim = 8
            self.observation_space.append(spaces.Box(low=-10.0, high=+10.0, shape=(obs_dim,), dtype=np.float32))

        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
  
        self._reset_render()


    def reset(self):
        # reset world
        self.type.rest_world(self.world)

        self._reset_render()
   
        obs_n = []
        # self.agents = self.world.agents
        for agent in self.world.agents:
            obs_n.append(self.type.observation(agent,self.world))
        return obs_n


    def _set_action(self, action, agent):

        entity_pos = []
        agent.action.u = np.zeros(self.world.dim_p)
        action = [action]

  
        is_stationary = True 
        for i in range(1, 7):
            if action[0][i] != 0:
                is_stationary = False  
                break

        if is_stationary == False:
            agent.energy -= 1


        agent.action.u[0] += action[0][1] - action[0][2]
        agent.action.u[1] += action[0][3] - action[0][4]
        agent.action.u[2] += action[0][5] - action[0][6]
        # entity_pos = self.type.observation(agent,self.world)
        # fai_z = math.asin(entity_pos[2] / math.sqrt(entity_pos[0] ** 2 + entity_pos[1] ** 2 + entity_pos[2] ** 2))
        # fai_z_fai = math.atan(entity_pos[1] / entity_pos[0])
        sensitity = 0.2
        agent.action.u *= sensitity
        # agent.action.u[0] *= (math.sin(fai_z)*math.cos(fai_z_fai))
        # agent.action.u[1] *= (math.sin(fai_z)*math.sin(fai_z_fai))
        # agent.action.u[2] *= (math.sin(fai_z))
        action = action[1:]
   
        assert len(action) == 0

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.agents

        for i, agent in enumerate(self.agents):
          
            self._set_action(action_n[i], agent)
    
        self.world.step()
        global all_entity_positions  
        global agent_velocity_data  
        global target_velocity_data  

        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
     
            entity_pos = obs_n[:3]  
            entity_distance = np.linalg.norm(entity_pos)  
            all_entity_positions[i].append(entity_distance)

            agent_velocity_magnitude = np.linalg.norm(agent.state.p_vel)
            target_velocity_magnitude = np.linalg.norm(agent.goal.state.p_vel)
      
            velocity_difference = abs(agent_velocity_magnitude - target_velocity_magnitude)
    
            agent_velocity_data[i].append(velocity_difference)

            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))

        for i, target in enumerate(self.targets):
         
            target_velocity = np.copy(target.state.p_vel)
            target_velocity_data[i].append(target_velocity)


        # reward = np.min(reward_n)
        # reward_n = [reward]*self.n

        return obs_n, reward_n,done_n,info_n

    def _get_obs(self, agent):
        # if self.observation_callback is None:
        #     return np.zeros(0)
        return self.type.observation(agent,self.world)

    def _get_reward(self, agent):
        # if self.reward_callback is None:
        #     return 0.0
        return self.type.reward(agent,self.world)

    def _get_info(self,agent):
        return {}

    def _get_done(self,agent):
        return False

    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            # for agent in self.world.agents:
            #     comm = []
            #     for other in self.world.agents:
            #         if other is agent: continue
            #         # if np.all(other.state.c == 0):
            #         #     word = '_'
            #         # else:
            #         #     word = alphabet[np.argmax(other.state.c)]
            #         message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
         
                self.viewers[i] = rendering.Viewer(1000, 1000)

   
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        # print(len(self.viewers))
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1.5
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(entity.state.p_pos[0], entity.state.p_pos[1])
            try:
                # render to display or array
                results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))
            except AttributeError as e:
                print("painting error",e)
                return 0

        return results


    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None
