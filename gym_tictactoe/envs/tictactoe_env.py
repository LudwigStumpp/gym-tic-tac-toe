import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .helpers import *


class TictactoeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    reward_violation = -100
    reward_normal = -1
    reward_win = 10

    def __init__(self):
        self.observation_space = spaces.Discrete(3 ** 9)
        self.action_space = spaces.Discrete(9 * 2)

    def step(self, action):
        player = 1 if action < 9 else 2
        done = False

        action_successful = self.turn(action)
        if not action_successful:
            reward = self.reward_violation
        else:
            if self.is_win(player):
                reward = self.reward_win
                done = True
            else:
                reward = self.reward_normal

        observation = self.s
        info = ''
        return observation, reward, done, info

    def reset(self):
        grid = [[0] * 3] * 3  # 3 times 3 grid
        self.s = self.encode(grid)

    def render(self, mode='human'):
        grid = self.decode(self.s)
        print_chars = [' ', 'O', 'X']

        rows = len(grid)
        cols = len(grid[0])

        for r in range(rows):
            for c in range(cols):
                print('|', end='')
                print(print_chars[grid[r][c]], end='')
            print('|')

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
        grid = list_to_array(base_3_rev, 3)

        return grid

    def turn(self, action):
        player = 1 if action < 9 else 2

        grid = self.decode(self.s)
        grid_flat = [item for sublist in grid for item in sublist]

        if grid_flat[action % 9] != 0:
            return False
        else:
            grid_flat[action % 9] = player
            new_grid = list_to_array(grid_flat, 3)
            self.s = self.encode(new_grid)
            return True

    def is_win(self, stone, num_winning=3):
        grid = self.decode(self.s)

        rows = len(grid)
        cols = len(grid[0])

        for r in range(rows):
            for c in range(cols):
                value = grid[r][c]
                if value == stone:

                    # left, top, right, bottom, top-left, top-right, bottom-right, bottom-left
                    check_ver_list = [0, -1, 0, 1, -1, -1, 1, 1]
                    check_hor_list = [-1, 0, 1, 0, -1, 1, 1, -1]

                    for i in range(len(check_ver_list)):
                        row_current = r
                        col_current = c

                        check_ver = check_ver_list[i]
                        check_hor = check_hor_list[i]

                        for line in range(num_winning - 1):
                            row_current = row_current + check_ver
                            col_current = col_current + check_hor

                            if row_current >= rows or col_current >= cols:
                                break

                            value_current = grid[row_current][col_current]
                            if value_current != stone:
                                break

                            if (line + 1) == (num_winning - 1):
                                return True

                    return False

        return False
