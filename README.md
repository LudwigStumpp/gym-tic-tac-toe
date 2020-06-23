# gym-tic-tac-toe

Open AI Gym environment for Tic-Tac-Toe.

## Installation

```
git clone https://github.com/LudwigStumpp/gym-tic-tac-toe.git
cd gym-tic-tac-toe
pip install -e .
```

## Usage

Make sure to have a look at the demo `gym-tic-tac-toe/main.py` where Q-Learning is used to train an agent.

### Initialization

```python
import gym

env = gym.make('tictactoe-v0')
env.reset()

env.render()
# | | | |
# | | | |
# | | | |
```

### Make a move

The board is indexed as follows:

```python
# |0|1|2|
# |3|4|5|
# |6|7|8|
```

To make a move, call the `step` - function:

```python
(observation, reward, done, info) = env.step([0, 3]) # 0 for player 1 and position 3
# (27, 0, False, 'normal move')

env.render()
# | | | |
# |O| | |
# | | | |

env.step([1, 2]) # 1 for player 2 and position 2
env.render()
# | | |X|
# |O| | |
# | | | |
```

### State representation
```python
preset = [[0, 1, 2], [0, 0, 0], [1, 0, 2]]
env.s = env._encode(preset)
env.render()
# | |O|X|
# | | | |
# |O| |X|

print(env.s) 
# 13872

board = env._decode(env.s)
# [[0, 1, 2], [0, 0, 0], [1, 0, 2]]
```

### 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests (`gym-tic-tac-toe/tests.py`) as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)