import gym
from gym import error, spaces, utils
import numpy as np
import random
import pickle
import os.path

import gym_tictactoe


def create_Q(env):
    if from_scratch:
        return np.zeros([env.observation_space.n, int(env.action_space.nvec[1])])
    else:
        try:
            print('Loading Q-Table')
            return pickle.load(open("q_table.p", "rb"))
        except IOError:
            print('Could not find file. Starting from scratch')
            return np.zeros([env.observation_space.n, int(env.action_space.nvec[1])])


def opponent_random(env):
    return [1, env.action_space.sample()[1]]


def opponent_random_better(env):
    valid_moves = env.get_valid_moves()
    return [1, random.choice(valid_moves)]


def opponent_human(env):
    action = [1, None]
    while action == [1, None] or not env.action_space.contains(action):
        print('Pick a move: ', end='')
        user_input = input()
        action[1] = int(user_input)-1 if user_input.isdigit() else None
    return action


def agent_move(action_space, state, Q, explore):
    if explore and random.uniform(0, 1) < epsilon:
        return [0, action_space.sample()[1]]  # explore action space
    else:
        return [0, np.argmax(Q[state])]  # exploit learned values


def play_one(env, Q, opponent, render=False, update=True, first=True, explore=True):
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
            action = agent_move(action_space, state, Q, explore)
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
            temp_diff = agent_reward + gamma * next_value - old_value
            Q[state, action] = old_value + alpha * temp_diff
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


def train(epochs, env, Q, opponents):
    is_first = True
    for i in range(epochs):
        (Q, _) = play_one(env, Q, random.choice(opponents), first=is_first)
        is_first = not is_first

    return Q


def test(epochs, env, Q, opponent):
    outcome_list = [None for i in range(epochs)]
    is_first = True
    for i in range(epochs):
        (_, outcome) = play_one(env, Q, opponent,
                                first=is_first, update=False, explore=False)
        outcome_list[i] = outcome
        is_first = not is_first

    wins = sum(1 for i in outcome_list if i == 'win')
    losses = sum(1 for i in outcome_list if i == 'loss')
    drawns = sum(1 for i in outcome_list if i == 'drawn')

    return wins, losses, drawns


def human_play(env, Q):
    is_first = True
    while True:
        print('New Game: Agent starts') if is_first else print(
            'New Game: You start')
        (_, outcome) = play_one(env, Q, opponent_human,
                                render=True, update=False, first=is_first, explore=False)
        if outcome == 'win':
            print('You lost')
        elif outcome == 'loss':
            print('You won')
        else:
            print('Drawn')

        is_first = not is_first


def main():
    # create environment
    env = gym.make('gym_tictactoe:tictactoe-v0',
                   size=size, num_winning=num_winning)

    # create empty Q-Table or preload
    Q = create_Q(env)

    # Play against a random player and learn
    print(f'Learning for {epochs} epochs')
    Q = train(epochs, env, Q, [opponent_random, opponent_random_better])
    print(f'Finished!')
    print('Saving Q-Table')
    pickle.dump(Q, open("q_table.p", "wb"))

    # Test
    print(f'Testing for {int(epochs/10)} epochs')
    wins, losses, drawns = test(int(epochs/10), env, Q, opponent_random_better)
    print(f'Wins: {wins}, Losses: {losses}, Drawns: {drawns}')

    # Play against human player
    human_play(env, Q)


# Hyperparameters
epsilon = 0.1  # exploration rate
alpha = 0.1  # learning rate
gamma = 0.8  # disount factor
epochs = 500000  # number of games played while training

# other
from_scratch = False

# Board settings
size = 3
num_winning = 3

main()
