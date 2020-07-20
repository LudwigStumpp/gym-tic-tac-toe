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

    def test_list_to_matrix(self):
        self.assertEqual(list_to_matrix([0] * 9, 3), [[0] * 3] * 3)


class TestGym(unittest.TestCase):
    def test_creation(self):
        custom_win_reward = 1000
        custom_normal_reward = -2
        custom_violation_reward = -5
        custom_drawn_reward = -1
        custom_size = 4

        env = gym.make('gym_tictactoe:tictactoe-v0',
                       reward_win=custom_win_reward,
                       reward_normal=custom_normal_reward,
                       reward_violation=custom_violation_reward,
                       reward_drawn=custom_drawn_reward,
                       size=custom_size)
        env.reset()

        # check custom rewards
        self.assertEqual(env.reward_win, custom_win_reward)
        self.assertEqual(env.reward_normal, custom_normal_reward)
        self.assertEqual(env.reward_violation, custom_violation_reward)
        self.assertEqual(env.reward_drawn, custom_drawn_reward)

        # check action and observation space
        self.assertEqual(env.action_space, spaces.MultiDiscrete(
            [2, custom_size * custom_size]))
        self.assertEqual(env.observation_space, spaces.Discrete(
            3 ** (custom_size * custom_size)))

    # from grid to decimal observation
    def test_encode(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')

        # 3 x 3
        self.assertEqual(env._encode([[0] * 3] * 3), 0)
        self.assertEqual(env._encode([[2] * 3] * 3), 19682)
        self.assertEqual(env._encode([[0, 0, 0], [0, 0, 0], [0, 0, 1]]), 6561)
        self.assertEqual(env._encode([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), 1)
        self.assertEqual(env._encode([[0, 2, 1], [2, 1, 1], [1, 2, 2]]), 18618)

        # 4 x 4
        self.assertEqual(env._encode([[0] * 4] * 4), 0)
        self.assertEqual(env._encode(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), 1)

    # from decimal observation to grid
    def test_decode(self):
        # 3 x 3
        env = gym.make('gym_tictactoe:tictactoe-v0')
        self.assertEqual(env._decode(0), [[0] * 3] * 3)
        self.assertEqual(env._decode(19682), [[2] * 3] * 3)
        self.assertEqual(env._decode(6561), [[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        self.assertEqual(env._decode(1), [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(env._decode(18618), [[0, 2, 1], [2, 1, 1], [1, 2, 2]])

        # 4 x 4
        env = gym.make('gym_tictactoe:tictactoe-v0', size=4)
        self.assertEqual(env._decode(0), [[0] * 4] * 4)
        self.assertEqual(env._decode(1), [[1, 0, 0, 0], [
                         0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

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
        env = gym.make('gym_tictactoe:tictactoe-v0')  # 3x3
        env.reset()

        self.assertEqual(env._turn([0, 0]), True)
        self.assertEqual(env._decode(env.s), [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(env._turn([0, 0]), False)
        self.assertEqual(env._turn([1, 0]), False)

        self.assertEqual(env._turn([1, 1]), True)
        self.assertEqual(env._decode(env.s), [[1, 2, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(env._turn([1, 1]), False)
        self.assertEqual(env._turn([0, 1]), False)

    def test_render(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        env.reset()
        env._turn([0, 0])
        env._turn([1, 1])

        with patch('sys.stdout', new=StringIO()) as fakeOutput:
            env.render()
            self.assertEqual(fakeOutput.getvalue().strip(),
                             '|O|X| |\n| | | |\n| | | |')

    def test_is_win(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        env.reset()
        self.assertEqual(env._is_win(1), False)
        self.assertEqual(env._is_win(0), True)

        env._turn([0, 0])
        env._turn([0, 1])
        env._turn([0, 2])
        self.assertEqual(env._is_win(2), False)
        self.assertEqual(env._is_win(1), True)

        env.reset()
        env._turn([1, 0])
        env._turn([1, 1])
        env._turn([1, 2])
        self.assertEqual(env._is_win(1), False)
        self.assertEqual(env._is_win(2), True)

        env.reset()
        env._turn([0, 0])
        env._turn([0, 3])
        env._turn([0, 6])
        self.assertEqual(env._is_win(1), True)

        env.reset()
        env._turn([0, 0])
        env._turn([0, 4])
        env._turn([0, 8])
        self.assertEqual(env._is_win(1), True)

        env.s = 8260
        self.assertEqual(env._is_win(2), True)

        env.s = env._encode([[1, 0, 0], [0, 1, 1], [0, 1, 0]])
        self.assertEqual(env._is_win(1), False)

        env.s = env._encode([[1, 2, 1], [2, 1, 1], [1, 2, 2]])
        self.assertEqual(env._is_win(1), True)

    def test_is_full(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        env.reset()

        env.s = env._encode([[0]*3]*3)
        self.assertEqual(env._is_full(), False)

        env.s = env._encode([[1]*3]*3)
        self.assertEqual(env._is_full(), True)

        env.s = env._encode([[2]*3]*3)
        self.assertEqual(env._is_full(), True)

    def test_step(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        env.reset()

        # normal move
        (observation, reward, done, info) = env.step([0, 0])
        self.assertEqual(env._decode(observation), [[1, 0, 0], [0]*3, [0]*3])
        self.assertEqual(reward, env.reward_normal)
        self.assertEqual(done, False)
        self.assertEqual(info, 'normal move player 1')

        # violation move
        (observation, reward, done, info) = env.step([0, 0])
        self.assertEqual(env._decode(observation), [[1, 0, 0], [0]*3, [0]*3])
        self.assertEqual(reward, env.reward_violation)
        self.assertEqual(done, False)
        self.assertEqual(info, 'invalid move player 1')

        # winning move
        env.step([0, 1])
        (observation, reward, done, info) = env.step([0, 2])
        self.assertEqual(env._decode(observation), [[1, 1, 1], [0]*3, [0]*3])
        self.assertEqual(reward, env.reward_win)
        self.assertEqual(done, True)
        self.assertEqual(info, 'winning move player 1')

        # drawn move
        env.s = env._encode([[0, 2, 1], [2, 1, 1], [2, 1, 2]])
        (observation, reward, done, info) = env.step([0, 0])
        self.assertEqual(env._decode(observation), [
                         [1, 2, 1], [2, 1, 1], [2, 1, 2]])
        self.assertEqual(reward, env.reward_drawn)
        self.assertEqual(done, True)
        self.assertEqual(info, 'drawn move player 1')

    def test_get_valid_moves(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')
        env.reset()
        env.step([0, 0])
        env.step([0, 8])
        env.step([1, 1])

        self.assertEqual(env.get_valid_moves(), [2, 3, 4, 5, 6, 7])


if __name__ == '__main__':
    unittest.main()
