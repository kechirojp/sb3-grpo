# example.py

import gymnasium as gym
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

# Import GRPO from the package
from sb3_grpo import GRPO

# --- 1. Define reward function for GRPO ---
# The core of GRPO is the ability to inject custom reward functions.
# Here we define a function that evaluates how "good" the next state is.
def cartpole_reward_fn(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    """
    Reward function for CartPole environment.
    Evaluates how "good" the next_state is.
    - Higher reward for pole angle closer to vertical
    - Higher reward for cart position closer to center
    """
    # next_state contents: [cart_pos, cart_vel, pole_angle, pole_vel]
    cart_pos = next_state[:, 0]
    pole_angle = next_state[:, 2]
    
    # Reward is higher when angle and position are closer to 0
    reward = 1.0 - torch.abs(pole_angle) - 0.1 * torch.abs(cart_pos)
    
    return reward.unsqueeze(-1)


# --- 2. Environment setup ---
# Standard Stable Baselines3 environment preparation
env = gym.make("CartPole-v1")
env = DummyVecEnv([lambda: env])


# --- 3. Create GRPO agent ---
# Usage is almost identical to PPO instantiation.
agent = GRPO(
    "MlpPolicy",
    env,
    reward_function=cartpole_reward_fn,  # Inject reward function here
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    verbose=1,
)

# --- 4. Training ---
# Just call the `learn` method like standard SB3 PPO
print("--- Starting GRPO Training ---")
agent.learn(total_timesteps=20000)
print("--- Training Finished ---")


# --- 5. Evaluate trained agent ---
print("\n--- Evaluating Trained Agent ---")
eval_env = gym.make("CartPole-v1")
obs, _ = eval_env.reset()
total_reward = 0
for _ in range(1000):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward
    if terminated or truncated:
        print(f"Episode finished with total reward: {total_reward}")
        total_reward = 0
        obs, _ = eval_env.reset()
eval_env.close()
