import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .helpers import *


class TictactoeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.observation_space = spaces.Discrete(3 ** 9)
        self.action_space = spaces.Discrete(9 * 2)

    def step(self, action):
        # return observation, reward, done, info
        ...

    def reset(self):
        grid = [[0] * 3] * 3  # 3 times 3 grid
        self.s = self.encode(grid)

    def render(self, mode='human'):
        print('Render Test')

    def close(self):
        ...

    def encode(self, grid):
        grid_flat = [item for sublist in grid for item in sublist]
        grid_flat_rev = list(reversed(grid_flat))

        return base_x_to_dec(grid_flat_rev, 3)

    def decode(self, dec):
        base_3 = dec_to_base_x(dec, 3)

        while len(base_3) < 9:
            base_3.insert(0, 0)

        base_3_rev = list(reversed(base_3))
        grid = [base_3_rev[r::3] for r in range(3)]

        return grid
