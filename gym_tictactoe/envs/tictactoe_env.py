import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .helpers import *


class TictactoeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=3, num_winning=3, reward_normal=0, reward_win=10, reward_violation=0, reward_drawn=0):
        self.num_winning = num_winning
        self.size = size
        self.num_fields = size**2
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(self.num_fields * 2)

        # rewards
        self.reward_normal = reward_normal
        self.reward_win = reward_win
        self.reward_violation = reward_violation
        self.reward_drawn = reward_drawn

    def step(self, action):
        player = 1 if action < self.num_fields else 2
        done = False
        info = ''

        action_successful = self._turn(action)
        if not action_successful:
            info = 'invalid move'
            reward = self.reward_violation
        else:
            if self._is_full():
                done = True

                if self._is_win(player):
                    info = 'winning move'
                    reward = self.reward_win
                else:
                    info = 'drawn move'
                    reward = self.reward_drawn

            else:
                if self._is_win(player):
                    info = 'winning move'
                    reward = self.reward_win
                    done = True
                else:
                    info = 'normal move'
                    reward = self.reward_normal

        observation = self.s
        return observation, reward, done, info

    def reset(self):
        grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.s = self._encode(grid)
        return self.s

    def render(self, mode='human'):
        grid = self._decode(self.s)
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

    def get_valid_moves(self):
        grid = self._decode(self.s)
        grid_flattened = [item for sublist in grid for item in sublist]
        return [i for i in range(len(grid_flattened)) if grid_flattened[i] == 0]

    # grid to dec
    def _encode(self, grid):
        grid_flat = [item for sublist in grid for item in sublist]
        grid_flat_rev = list(reversed(grid_flat))

        return base_x_to_dec(grid_flat_rev, 3)

    # dec to grid
    def _decode(self, dec):
        base_3 = dec_to_base_x(dec, 3)

        while len(base_3) < self.num_fields:
            base_3.insert(0, 0)

        base_3_rev = list(reversed(base_3))
        grid = list_to_matrix(base_3_rev, self.size)

        return grid

    def _turn(self, action):
        player = 1 if action < self.num_fields else 2

        grid = self._decode(self.s)
        grid_flat = [item for sublist in grid for item in sublist]

        if grid_flat[action % self.num_fields] != 0:
            # invalid move
            return False
        else:
            # valid move
            grid_flat[action % self.num_fields] = player
            new_grid = list_to_matrix(grid_flat, self.size)
            self.s = self._encode(new_grid)
            return True

    def _is_win(self, stone):
        grid = self._decode(self.s)

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

                        for line in range(self.num_winning - 1):
                            row_current = row_current + check_ver
                            col_current = col_current + check_hor

                            if row_current >= rows or col_current >= cols or row_current < 0 or col_current < 0:
                                break

                            value_current = grid[row_current][col_current]
                            if value_current != stone:
                                break

                            if (line + 1) == (self.num_winning - 1):
                                return True

        return False

    def _is_full(self):
        grid = self._decode(self.s)
        grid_flat = [item for sublist in grid for item in sublist]
        if sum(1 for i in grid_flat if i != 0) == len(grid_flat):
            return True
        return False
