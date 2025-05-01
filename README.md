# Chess-RL

This project implements a reinforcement learning agent that learns to play chess using a custom environment built with Gymnasium and trained using the PPO algorithm from Stable-Baselines3.

## Project Structure

- `chess_env.py`: Custom Gym-compatible chess environment using `python-chess`
- `train.py`: Script to train the PPO agent
- `play.py`: Script to run the trained agent and display moves in the terminal
- `requirements.txt`: Python dependencies

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/anjaanasnambiar/chess-rl.git 
```

2. Install dependencies:

```bash
pip install -r requirements.txt

```

3. Training the Agent

To train the chess-playing agent using PPO:

``` bash

python train.py

```

This will save a trained model to ppo_chess_agent.zip.


4. Playing a Game

To watch the trained agent play a game:

``` bash
python play.py
```

This will print the board state in the terminal after each move until the game ends.

## Environment Details
    Observation Space: Box(8, 8, 12)
    Represents the board as 12 binary layers (bitboards) — 6 piece types × 2 colors.
    Action Space: Discrete(4672)

Encodes all possible UCI moves in a fixed-size list, padded for compatibility.

## Reward Function

    +1 for winning (checkmate)
    0.5 for drawing (stalemate or insufficient material)
    -1 for making an illegal move

## Dependencies
The environment requires:

      Python 3.8 or higher
      gymnasium
      stable-baselines3
      torch
      python-chess
      numpy

## Install them using

``` bash
pip install -r requirements.txt
```

## License
This project is open-source and available under the MIT License.

