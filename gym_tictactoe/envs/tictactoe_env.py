import gym
from gym import error, spaces, utils
from gym.utils import seeding


class TictactoeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # self.observation_space =
        # self.action_space =
        ...

    def step(self, action):
        # return observation, reward, done, info
        ...

    def reset(self):
        ...

    def render(self, mode='human'):
        print('Render Test')

    def close(self):
        ...
