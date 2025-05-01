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
cd chess-rl
