import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

    
class OptimizationEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    NMBR_STATES = 2
    NMBR_ACTIONS = NMBR_STATES

    def __init__(self, reset_params=False, continuous=False):
        super(OptimizationEnv, self).__init__()
        self.action_space = spaces.Box(low=- np.ones(self.NMBR_STATES),
                                       high=np.ones(self.NMBR_STATES),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=- np.ones(self.NMBR_STATES),
                                       high=np.ones(self.NMBR_STATES),
                                       dtype=np.float32)
        self.state = np.random.uniform(-1,1,size=self.NMBR_STATES)
        self.nmbr_minima = 1
        self.params = self.make_params()
        self.reset_params = reset_params
        self.continuous = continuous
        
    def make_params(self):
        return {'x0': np.random.uniform(-1,1,size=(self.nmbr_minima,2)), 
                'k': np.random.normal(loc=1, scale=.2,size=self.nmbr_minima), 
                'noise': np.random.uniform(0,.01)}
        
    def loss(self, x, y):
        loss = 0
        for x0,y0,k in zip(self.params['x0'][:,0], self.params['x0'][:,1], self.params['k']):
            loss += k * ((x - x0)**2 + (y - y0)**2)
            loss += np.random.normal(scale=self.params['noise'],size=loss.shape)
        return loss
        
    def step(self, action):
        
        info = {}
        
        new_state = action
        reward = - self.loss(new_state[0], new_state[1])
        terminated = True if np.linalg.norm(self.params['x0'] - new_state) < 0.1 and not self.continuous else False
        if terminated:
            reward += 1
        truncated = False
        
        self.state = new_state
        
        return new_state, reward, terminated, truncated, info
    
    def reset(self, state=None, new_params=None):
        info = {}
        if state is None:
            self.state = np.random.uniform(-1,1,size=self.NMBR_STATES)
        else:
            self.state = state
        if new_params or (new_params is None and self.reset_params):
            self.params = self.make_params()
        return self.state, info
        
    def render(self):
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)

        X, Y = np.meshgrid(x, y)
        Z = self.loss(X, Y)

        plt.contour(X, Y, Z, levels=10, colors='black')
        plt.scatter(self.state[0], self.state[1], color='red', s=100)
        plt.scatter(self.params['x0'][:,0], self.params['x0'][:,1], color='green', s=100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()