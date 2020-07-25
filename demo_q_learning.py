import gym
import numpy as np
import random
import pickle
import os.path

import gym_tictactoe


def create_Q(env):
    """
    Initializes Q-Table, where:
        rows = states
        columns = actions
        entries = values = sum of accumulated expected reward

    Args:
        env: The environment to create the Q-table for

    Returns:
        zero-matrix m x n where:
            m = observation space
            n = action space
    """

    if from_scratch:
        return np.zeros([env.observation_space.n, int(env.action_space.nvec[1])])
    else:
        try:
            print('Loading Q-Table')
            return pickle.load(open("q_table.p", "rb"))
        except IOError:
            print('Could not find file. Starting from scratch')
            return np.zeros([env.observation_space.n, int(env.action_space.nvec[1])])


def get_next_envs(env, turn):
    """
    Get list of possible next environments given a player at turn.

    Args:
        env: The environment to create the Q-table for
        turn: {0, 1}, which player is at turn

    Returns:
        list of possible gym_tictactoe.envs.tictactoe_env.TictactoeEnv
            by looking at all possible moves given the player at turn
    """

    free_moves = env.get_valid_moves()
    next_envs = []
    for free_move in free_moves:
        env_copy = gym.make(
            'gym_tictactoe:tictactoe-v0', size=size, num_winning=num_winning)
        env_copy.s = env.s
        env_copy.step((turn, free_move))
        next_envs.append(env_copy)
    return (next_envs, free_moves)


def minmax_state_value(env, player, turn):
    """
    Calculates the value of a board state given a player's view and a player at turn.
    Makes use of minmax algorithm.

    Args:
        env: The environment
        player: {0, 1}, the player's view
        turn: {0, 1}, which player is at turn

    Returns:
        the value of the current state of the environment in the eyes of a player.
            int {-1, 0, 1} where:
                1 is best value
                -1 is worst value
    """

    # end cases
    if env._is_win(player+1):
        return 1
    elif env._is_win(int(not player)+1):
        return -1
    elif env._is_full():
        return 0

    # build possible next boards given which player is at turn
    (next_envs, _) = get_next_envs(env, turn)

    combine_func = max if turn == player else min
    return combine_func([minmax_state_value(next_env, player, int(not turn)) for next_env in next_envs])


def opponent_minmax(env, player=1):
    """
    Opponent for the agent that chooses the move that will result in the next state with the highest value
    given by a minmax algorithm.

    Args:
        env: The environment
        player=1: {0, 1} for which player to decide

    Returns:
        action to take in form (a, b) where:
            a = player = 1
            b = field to place stone by index
    """
    (next_envs, free_moves) = get_next_envs(env, player)
    return (player, free_moves[np.argmax([minmax_state_value(next_env, player, int(not player)) for next_env in next_envs])])


def opponent_random(env, player=1):
    """
    Opponent for the agent that chooses a random position on the board.
    Might result in invalid moves which will have no effect

    Args:
        env: The environment
        player=1: {0, 1} for which player to decide

    Returns:
        action to take in form (a, b) where:
            a = player
            b = field to place stone by index
    """

    return (player, env.action_space.sample()[1])


def opponent_random_better(env, player=1):
    """
    Opponent for the agent that chooses a random free position on the board.
    Therefore only chooses a valid action

    Args:
        env: The environment
        player=1: {0, 1} for which player to decide

    Returns:
        action to take in form (a, b) where:
            a = player
            b = field to place stone by index
    """

    valid_moves = env.get_valid_moves()
    return (player, random.choice(valid_moves))


def opponent_human(env, player=1):
    """
    Human opponent. Asks for input via terminal until a valid action is transmitted

    Args:
        env: The environment
        player=1: {0, 1} for which player to decide

    Returns:
        action to take in form (a, b) where:
            a = player
            b = field to place stone by index
    """

    action = [player, None]
    while action == [player, None] or not env.action_space.contains(action):
        print('Pick a move: ', end='')
        user_input = input()
        action[1] = int(user_input)-1 if user_input.isdigit() else None
    return tuple(action)


def agent_move(action_space, state, Q, explore, player=0):
    """
    Agent move decision.
    Given the state and the values of the Q table, agent chooses action with maximum value.
    Chance to also ignore Q-table and to explore new actions. 

    Args:
        action_space: action space of the environment
        state: current observed state of the environment
        Q: Q-table for exploitation
        explore: True/False to tell if able to explore

    Returns:
        action to take in form (a, b) where:
            a = player
            b = field to place stone by index
    """

    if explore and random.uniform(0, 1) < epsilon:
        return (player, action_space.sample()[1])  # explore action space
    else:
        return (player, np.argmax(Q[state]))  # exploit learned values


def play_one(env, Q, opponent, render=False, update=True, first=True, explore=True):
    """
    Agent plays one match against an opponent.

    Args:
        env: The environment
        Q: Q-table
        opponent: function (env) -> action, returns action given an environment
        render=False: Whether to display the field after each move
        update=True: Whether to update the Q-table
        first=True: If agent starts the game
        explore=True: If exploration is allowed for agent decision making

    Returns:
        Tuple of (a, b) where
            a = (updated) Q-Table
            b = outcome of the move = {None, 'win', 'loss', 'drawn'}
    """

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

    # Game finished, get outcome for agent
    outcome = None
    if env._is_win(1):
        outcome = 'win'
    elif env._is_win(2):
        outcome = 'loss'
    else:
        outcome = 'drawn'

    return (Q, outcome)


def train(epochs, env, Q, opponents):
    """
    Train the agent

    Args:
        epochs: Number of epochs to train
        env: The environment
        Q: Q-table
        opponents: list of opponents as functions (env) -> action to play against

    Returns:
        updated Q-table
    """

    is_first = True
    for i in range(epochs):
        (Q, _) = play_one(env, Q, random.choice(opponents), first=is_first)
        is_first = not is_first

    return Q


def test(epochs, env, Q, opponent, render=False):
    """
    Evaluate the performance of the agent by playing against one opponent.
    No exploration, no updates to Q-table.

    Args:
        epochs: Number of epochs to train
        env: The environment
        Q: Q-table
        opponent: opponent as function (env) -> action to play against
        render=False: If board shall be rendered and game outcome printed

    Returns:
        Tuple (wins, losses, drawns)
    """

    outcome_list = [None for i in range(epochs)]
    is_first = True
    for i in range(epochs):
        (_, outcome_agent) = play_one(env, Q, opponent, render=render,
                                      first=is_first, update=False, explore=False)

        if render:
            if outcome_agent == 'win':
                print('You lost')
            elif outcome_agent == 'loss':
                print('You won')
            else:
                print('Drawn')

        outcome_list[i] = outcome_agent
        is_first = not is_first

    wins = sum(1 for i in outcome_list if i == 'win')
    losses = sum(1 for i in outcome_list if i == 'loss')
    drawns = sum(1 for i in outcome_list if i == 'drawn')

    return wins, losses, drawns


def main():
    """
    Main function:
        1. Create environment
        2. Preload Q-table if present, otherwise create from scratch
        3. Train for n epochs playing against random opponent
        4. Test for n/10 epochs and show performance
        5. Play against human

    Args:
        -

    Returns:
        -
    """

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
    print(f'Playing 4 games against human player')
    losses, wins, drawns = test(4, env, Q, opponent_human, render=True)
    print(f'Wins: {wins}, Losses: {losses}, Drawns: {drawns}')


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

if __name__ == '__main__':
    main()
