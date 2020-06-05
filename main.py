import gym
from gym import error, spaces, utils
import numpy as np
import random

import gym_tictactoe


def player_random(env):
    return env.action_space.sample() % 9 + 9


def player_human(env):
    print('Pick a move: ', end='')
    return int(input()) + 9  # 0 - 8


def player_agent(action_space, state, Q):
    if random.uniform(0, 1) < epsilon:
        return action_space.sample() % 9  # explore action space
    else:
        return np.argmax(Q[state])  # exploit learned values


def play(env, Q, epochs, opponent, render=False, update=True):
    action_space = env.action_space

    for i in range(epochs):
        state = env.reset()

        done = False

        while not done:
            # Agent makes move
            action = player_agent(action_space, state, Q)
            (next_state, agent_reward, done, info) = env.step(action)
            old_value = Q[state, action]

            [print('Agent moved:'),
             env.render(), print()] if render else None

            # Opponent makes a move, but only if not done
            if not done:
                opponent_action = opponent(env)
                (next_state, opponent_reward, done,
                 info) = env.step(opponent_action)

                [env.render(), print()] if render else None
            else:
                opponent_reward = 0

            # update Q Table but only after opponent has moved
            if update:
                agent_reward -= opponent_reward
                next_value = np.max(Q[next_state])
                temp_diff = agent_reward + gamma * next_value
                Q[state, action] = old_value + alpha * (temp_diff - old_value)

            state = next_state

    return Q


# Hyperparameters
epsilon = 0.1  # exploration rate
alpha = 0.1  # learning rate
gamma = 0.6  # disount factor


def main():
    global epsilon
    # create environment
    env = gym.make('gym_tictactoe:tictactoe-v0')

    # create Q-Table
    Q = np.zeros([env.observation_space.n, int(env.action_space.n / 2)])

    # Play against a random player and learn
    Q = play(env, Q, 1000000, player_random)

    # Play against human player
    epsilon = 0  # do not explore
    play(env, Q, 100, player_human, render=True, update=False)


main()
