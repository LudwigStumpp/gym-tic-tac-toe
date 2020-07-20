import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .helpers import *


class TictactoeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=3, num_winning=3, reward_normal=0, reward_win=10, reward_violation=0, reward_drawn=0):
        """
        Initializes an Tic-Tac-Toe Open AI gym environment.
        Make sure to call the reset function to reset to an empty space before making a move.

        Squared board of a given size with indexed positions:
            e.g. 3x3
            [0-1-2]
            [3-4-5]
            [6-7-8]

        State space is given by the total number of possible states = 3^number_fields
        Action space is Multi-Discrete [2, size^2] where the first index declares the player and the second the board position to make a move
            e.g Player 1 on the middle of the board: [0, 4]
            Player 2 on the bottom left corner: [1, 6]

        Args:
            size=3: Size of the board, size X size
            num_winning=3: Number of equal stones to win the game
            reward_normal=0: Reward for a standard valid move
            reward_win=10: Reward for winning
            reward_violation=0: Reward if invalid move, move on already placed position
            reward_drawn=0: Reward for Drawn

        Returns:
            -
        """

        self.num_winning = num_winning
        self.size = size
        self.num_fields = size**2
        self.observation_space = spaces.Discrete(3 ** self.num_fields)
        self.action_space = spaces.MultiDiscrete([2, self.num_fields])

        # rewards
        self.reward_normal = reward_normal
        self.reward_win = reward_win
        self.reward_violation = reward_violation
        self.reward_drawn = reward_drawn

    def step(self, action):
        """
        Performs action on given state to get reward and observation

        Args:
            action: This is the action to take, must be inside the action space.
                Format (player, position) where player = {0, 1} and position = [0; fields-1]

            Given a 3x3 board, a valid action would be (0, 8) for player 1 on the last field in the bottom right corner

        Returns:
            A tuple (observation, reward, done, info):

            observation: new encoded state of the environment
            reward: numeric reward
            done: True/False, if the game is finished. True if action resulted in a win
            info: Description of the action
        """

        player = action[0] + 1
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
        return observation, reward, done, info + f' player {player}'

    def reset(self):
        """
        Resets the environment to the beginning state where the board is empty

        Args:
            -

        Returns:
            -
        """

        grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.s = self._encode(grid)
        return self.s

    def render(self):
        """
        Function to render the current board state and display it.

        Args:
            -

        Returns:
            -
        """

        grid = self._decode(self.s)
        print_chars = [' ', 'O', 'X']

        rows = len(grid)
        cols = len(grid[0])

        for r in range(rows):
            for c in range(cols):
                print('|', end='')
                print(print_chars[grid[r][c]], end='')
            print('|')

    def get_valid_moves(self):
        """
        Returns a list of possible moves to make by indices on the board

        Args:
            -

        Returns:
            List of indices that represent free board positions as indices starting by 0
        """

        grid = self._decode(self.s)
        grid_flattened = [item for sublist in grid for item in sublist]
        return [i for i in range(len(grid_flattened)) if grid_flattened[i] == 0]

    # grid to dec
    def _encode(self, grid):
        """
        Encodes a grid representation into a decimal identifying value.

        Args:
            grid: 2-D Array representation of the board where:
                0 = free position
                1 = Player 1
                2 = Player 2 

        Returns:
            Encoded number representing the state
        """

        grid_flat = [item for sublist in grid for item in sublist]
        grid_flat_rev = list(reversed(grid_flat))

        return base_x_to_dec(grid_flat_rev, 3)

    # dec to grid
    def _decode(self, dec):
        """
        Decodes a board state into a 2-D grid representation.

        Args:
            dec: Decimal state encoding.

        Returns:
            2-D Array representation of the board where:
                0 = free position
                1 = Player 1
                2 = Player 2 
        """

        base_3 = dec_to_base_x(dec, 3)

        while len(base_3) < self.num_fields:
            base_3.insert(0, 0)

        base_3_rev = list(reversed(base_3))
        grid = list_to_matrix(base_3_rev, self.size)

        return grid

    def _turn(self, action):
        """
        Placing a stone on the board given an action.

        Args:
            action: Action to take, must be withing action space.

        Returns:
            True/False given a move was valid or not. Move is invalid if position is not empty
        """

        player = action[0] + 1
        place = action[1]

        grid = self._decode(self.s)
        grid_flat = [item for sublist in grid for item in sublist]

        if grid_flat[place] != 0:
            # invalid move
            return False
        else:
            # valid move
            grid_flat[place] = player
            new_grid = list_to_matrix(grid_flat, self.size)
            self.s = self._encode(new_grid)
            return True

    def _is_win(self, player):
        """
        Checks is winning state for player given the attribute num_winning that is optinally passed on init.

        Args:
            player: Player to check, either 1 or 2

        Returns:
            True if player has won, otherwise false
        """

        grid = self._decode(self.s)

        rows = len(grid)
        cols = len(grid[0])

        for r in range(rows):
            for c in range(cols):
                value = grid[r][c]
                if value == player:

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
                            if value_current != player:
                                break

                            if (line + 1) == (self.num_winning - 1):
                                return True

        return False

    def _is_full(self):
        """
        Checks if full board.

        Args:
            -

        Returns:
            True if board is full, otherwise False
        """

        grid = self._decode(self.s)
        grid_flat = [item for sublist in grid for item in sublist]
        if sum(1 for i in grid_flat if i != 0) == len(grid_flat):
            return True
        return False
