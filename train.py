from chess_env import ChessEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = ChessEnv()
check_env(env)  

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

model.save("ppo_chess_agent")
