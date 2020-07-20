import gym
import gym_tictactoe
import itertools


def main():
    env = gym.make('gym_tictactoe:tictactoe-v0')

    observation = env.reset()
    while True:
        player = next_player()
        position = env.action_space.sample()[1]
        observation, reward, done, info = env.step((player, position))
        env.render()

        if done:
            print(info)
            break


next_player = itertools.cycle([0, 1]).__next__

if __name__ == '__main__':
    main()
