import gym
from gym import error, spaces, utils
import numpy as np
import random
import pickle
import os.path

import gym_tictactoe


def opponent_random(env):
    return env.action_space.sample() % 9 + 9


def opponent_human(env):
    action = None
    while action not in list(range(9, 18)):
        print('Pick a move: ', end='')
        user_input = input()
        action = int(user_input) + 9 - 1 if user_input.isdigit() else None

    return action


def agent_move(action_space, state, Q):
    if random.uniform(0, 1) < epsilon:
        return action_space.sample() % 9  # explore action space
    else:
        return np.argmax(Q[state])  # exploit learned values


def play_one(env, Q, opponent, render=False, update=True, first=True):
    action_space = env.action_space

    state = env.reset()

    done = False
    agent_moved = False
    opponent_moved = False
    agent_reward = 0
    old_value = None

    # Play
    while not done:
        # Agent moves, skip in first round if second
        if first or opponent_moved:
            action = agent_move(action_space, state, Q)
            (next_state, agent_reward, done, info) = env.step(action)
            agent_moved = True
            old_value = Q[state, action]

            [print('Agent moved:'),
                env.render(), print()] if render else None

        # Opponent makes a move, but only if not done
        if not done:
            opponent_action = opponent(env)
            (next_state, opponent_reward, done,
                info) = env.step(opponent_action)
            opponent_moved = True

            [env.render(), print()] if render else None
        else:
            opponent_reward = 0

        # update Q Table but only after opponent has moved
        if update and agent_moved:
            agent_reward -= opponent_reward
            next_value = np.max(Q[next_state])
            temp_diff = agent_reward + gamma * next_value
            Q[state, action] = old_value + alpha * (temp_diff - old_value)
        state = next_state

    # Game finished
    outcome = None
    if env._is_win(1):
        outcome = 'win'
    elif env._is_win(2):
        outcome = 'loss'
    else:
        outcome = 'drawn'

    return (Q, outcome)


# Hyperparameters
epsilon = 0.15  # exploration rate
alpha = 0.1  # learning rate
gamma = 0.6  # disount factor
epochs = 100000  # number of games played while training
from_scratch = False


def main():
    # create environment
    env = gym.make('gym_tictactoe:tictactoe-v0')

    # create empty Q-Table or preload
    if from_scratch:
        Q = np.zeros([env.observation_space.n, int(env.action_space.n / 2)])
    else:
        try:
            print('Loading Q-Table')
            Q = pickle.load(open("q_table.p", "rb"))
        except IOError:
            print('Could not find file. Starting from scratch')
            Q = np.zeros([env.observation_space.n,
                          int(env.action_space.n / 2)])

    # Play against a random player and learn
    print(f'Learning for {epochs} epochs')
    for i in range(int(epochs)):
        (Q, _) = play_one(env, Q, opponent_random, first=True)
        (Q, _) = play_one(env, Q, opponent_random, first=False)
    print(f'Finished!')

    print('Saving Q-Table')
    pickle.dump(Q, open("q_table.p", "wb"))

    # Play against human player
    global epsilon
    epsilon = 0  # do not explore
    is_first = True
    while True:
        print('New Game: Agent starts') if is_first else print(
            'New Game: You start')
        (_, outcome) = play_one(env, Q, opponent_human,
                                render=True, update=False, first=is_first)
        if outcome == 'win':
            print('You lost')
        elif outcome == 'loss':
            print('You won')
        else:
            print('Drawn')


main()
