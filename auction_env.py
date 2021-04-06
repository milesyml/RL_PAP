import numpy as np
from gym import Env, spaces
from gym.utils import seeding

class HighLowEnv(Env):
    """Custom Environment that follows gym interface"""
    def __init__(self, v_high=1, v_low=0.3, p_high=0.8, p_low=0.1, distribution='uniform', episode_length=1000):
        super(HighLowEnv, self).__init__()

        # Set Agent Type 
        assert distribution.lower() in ['uniform', 'exponential', 'normal']
        self.distribution = distribution.lower()
        self.draw_new_agent()

        # Set utility and work parameters
        self.v_high = v_high
        self.v_low = v_low
        self.p_high = p_high
        self.p_low = p_low
        self.time_step = 0
        self.episode_length = episode_length

        # Set reward function (Action Space)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Change to 2 states, start and end
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    def step(self, action):
        self.draw_new_agent()

        low_reward, high_reward = action
        low_reward = self.inverse_scale(low_reward)
        high_reward = self.inverse_scale(high_reward)      

        low_u = self.p_low*high_reward + (1-self.p_low)*low_reward
        high_u = self.p_high*high_reward + (1-self.p_high)*low_reward - self.type

        if high_u >= low_u:
            outcome_prob = [1-self.p_high, self.p_high]
        else:
            outcome_prob = [1-self.p_low, self.p_low]

        outcome = str(np.random.choice(['low','high'], p=outcome_prob))

        if outcome == 'high':
            reward = self.v_high - high_reward
        else:
            reward = self.v_low - low_reward

        if self.time_step >= self.episode_length:
            done = True
        else:
            done = False
            self.time_step += 1

        observation = np.array([1])
        info = {'type': self.type, 'reward': action}
        return observation, reward, done, info

    def reset(self):
        # Draw new agent when reset
        self.time_step = 0
        return np.array([0]) # reward, done, info can't be included

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def draw_new_agent(self):
        if self.distribution == 'uniform':
            self.type = np.random.uniform()

        elif self.distribution == 'exponential':
            self.type = np.random.exponential()

        elif self.distribution == 'normal':
            self.type = np.random.normal()

        return

    @staticmethod
    def inverse_scale(action):
        return (action+1)/2        

class ContEnv(Env):
    """Custom Environment that follows gym interface"""
    def __init__(self, episode_length=1000):
        super(ContEnv, self).__init__()

        # Set utility and work parameters
        self.time_step = 0
        self.episode_length = episode_length

        # Set reward function (Action Space)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Change to 2 states, start and end
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    def step(self, action):
        gradient, constant = action

        # Scaling
        gradient = self.inverse_scale(gradient)
        constant = (constant-1)/2        

        agent_type = self.draw_new_agent()
        effort = self.agent_effort(agent_type, gradient, constant)
        cost = self.agent_cost(agent_type, effort)

        if gradient*effort + constant - cost >= 0:
            outcome = self.agent_production(agent_type, effort)
            reward = outcome - gradient*outcome - constant
        else:
            outcome = 0
            reward = 0

        if self.time_step >= self.episode_length:
            done = True
        else:
            done = False
            self.time_step += 1

        observation = np.array([1])
        info = {'type': agent_type, 'reward_fn': action, 'outcome':outcome}
        return observation, reward, done, info

    def reset(self):
        # Draw new agent when reset
        self.time_step = 0
        return np.array([0]) # reward, done, info can't be included

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def draw_new_agent(self):
        return np.random.uniform(1,3)
    
    @staticmethod
    def agent_effort(t, gradient, constant):
        return gradient*t/2

    @staticmethod
    def agent_cost(t, effort):
        return effort**2/t

    @staticmethod
    def agent_production(t, effort):
        return effort + np.random.normal(0, t/100)    

    @staticmethod
    def inverse_scale(action):
        return (action+1)/2    