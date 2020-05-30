import unittest
import gym

import gym_tictactoe


class TestGym(unittest.TestCase):
    def test_creation(self):
        env = gym.make('gym_tictactoe:tictactoe-v0')


if __name__ == '__main__':
    unittest.main()
