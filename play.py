from stable_baselines3 import PPO
from chess_env import ChessEnv

model = PPO.load("ppo_chess_agent")
env = ChessEnv()

obs, _ = env.reset()
done = False

print("Starting a new game: ")
env.render() 

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    env.render()
    print("-" * 40)

if reward == 1:
    print("Agent won!")
elif reward == 0.5:
    print("Draw.")
elif reward == -1:
    print("Agent made an illegal move!")
else:
    print("Game ended.")
