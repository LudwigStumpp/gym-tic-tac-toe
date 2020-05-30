import unittest
import gym
from gym import error, spaces, utils

import gym_tictactoe
from gym_tictactoe.envs.helpers import *


class TestHelpers(unittest.TestCase):
    def test_base_x_to_dec(self):
        self.assertEqual(base_x_to_dec([1, 0], 2), 2)
        self.assertEqual(base_x_to_dec([1, 0, 0], 2), 4)
        self.assertEqual(base_x_to_dec([2, 0], 3), 6)

    def test_dec_to_base_x(self):
        self.assertEqual(dec_to_base_x(2, 2), [1, 0])
        self.assertEqual(dec_to_base_x(4, 2), [1, 0, 0])
        self.assertEqual(dec_to_base_x(6, 3), [2, 0])


class TestGym(unittest.TestCase):
    def test_creation(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')

        # check action and observation space
        self.assertEqual(env.action_space, spaces.Discrete(2 * 9))
        self.assertEqual(env.observation_space, spaces.Discrete(3 ** 9))

    # from grid to decimal observation
    def test_encode(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        self.assertEqual(env.encode([[0] * 3] * 3), 0)
        self.assertEqual(env.encode([[2] * 3] * 3), 19682)
        self.assertEqual(env.encode([[0, 0, 0], [0, 0, 0], [0, 0, 1]]), 6561)
        self.assertEqual(env.encode([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), 1)

    # from decimal observation to grid
    def test_decode(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        self.assertEqual(env.decode(0), [[0] * 3] * 3)
        self.assertEqual(env.decode(19682), [[2] * 3] * 3)
        self.assertEqual(env.decode(6561), [[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        self.assertEqual(env.decode(1), [[1, 0, 0], [0, 0, 0], [0, 0, 0]])

    def test_reset(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        env.reset()
        self.assertEqual(env.s, 0)

    def test_preset(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        state = [[0, 0, 0], [0, 0, 0], [0, 0, 1]]
        env.s = env.encode(state)
        self.assertEqual(env.s, 6561)


if __name__ == '__main__':
    unittest.main()
