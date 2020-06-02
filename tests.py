import unittest

# for testing print output
from unittest.mock import patch
from io import StringIO

# gym default
import gym
from gym import error, spaces, utils

# gym tic tac toe
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

    def test_list_to_array(self):
        self.assertEqual(list_to_array([0] * 9, 3), [[0] * 3] * 3)


class TestGym(unittest.TestCase):
    def test_creation(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')

        # check action and observation space
        self.assertEqual(env.action_space, spaces.Discrete(2 * 9))
        self.assertEqual(env.observation_space, spaces.Discrete(3 ** 9))

    # from grid to decimal observation
    def test_encode(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        self.assertEqual(env._encode([[0] * 3] * 3), 0)
        self.assertEqual(env._encode([[2] * 3] * 3), 19682)
        self.assertEqual(env._encode([[0, 0, 0], [0, 0, 0], [0, 0, 1]]), 6561)
        self.assertEqual(env._encode([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), 1)

    # from decimal observation to grid
    def test_decode(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        self.assertEqual(env._decode(0), [[0] * 3] * 3)
        self.assertEqual(env._decode(19682), [[2] * 3] * 3)
        self.assertEqual(env._decode(6561), [[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        self.assertEqual(env._decode(1), [[1, 0, 0], [0, 0, 0], [0, 0, 0]])

    def test_reset(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        env.reset()
        self.assertEqual(env.s, 0)

    def test_preset(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        state = [[0, 0, 0], [0, 0, 0], [0, 0, 1]]
        env.s = env._encode(state)
        self.assertEqual(env.s, 6561)

    def test_turn(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        env.reset()

        self.assertEqual(env._turn(0), True)
        self.assertEqual(env._decode(env.s), [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(env._turn(0), False)
        self.assertEqual(env._turn(9), False)

        self.assertEqual(env._turn(10), True)
        self.assertEqual(env._decode(env.s), [[1, 2, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(env._turn(10), False)
        self.assertEqual(env._turn(1), False)

    def test_render(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        env.reset()
        env._turn(0)
        env._turn(10)

        with patch('sys.stdout', new=StringIO()) as fakeOutput:
            env.render()
            self.assertEqual(fakeOutput.getvalue().strip(),
                             '|O|X| |\n| | | |\n| | | |')

    def test_is_win(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        env.reset()
        self.assertEqual(env._is_win(1), False)
        self.assertEqual(env._is_win(0), True)

        env._turn(0)
        env._turn(1)
        env._turn(2)
        self.assertEqual(env._is_win(2), False)
        self.assertEqual(env._is_win(1), True)

        env.reset()
        env._turn(9)
        env._turn(10)
        env._turn(11)
        self.assertEqual(env._is_win(1), False)
        self.assertEqual(env._is_win(2), True)

        env.reset()
        env._turn(0)
        env._turn(3)
        env._turn(6)
        self.assertEqual(env._is_win(1), True)

        env.reset()
        env._turn(0)
        env._turn(4)
        env._turn(8)
        self.assertEqual(env._is_win(1), True)

    def test_step(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        env.reset()

        # normal move
        (observation, reward, done, info) = env.step(0)
        self.assertEqual(env._decode(observation), [[1, 0, 0], [0]*3, [0]*3])
        self.assertEqual(reward, env.reward_normal)
        self.assertEqual(done, False)

        # violation move
        (observation, reward, done, info) = env.step(0)
        self.assertEqual(env._decode(observation), [[1, 0, 0], [0]*3, [0]*3])
        self.assertEqual(reward, env.reward_violation)
        self.assertEqual(done, False)

        # winning move
        env.step(1)
        (observation, reward, done, info) = env.step(2)
        self.assertEqual(env._decode(observation), [[1, 1, 1], [0]*3, [0]*3])
        self.assertEqual(reward, env.reward_win)
        self.assertEqual(done, True)


if __name__ == '__main__':
    unittest.main()
