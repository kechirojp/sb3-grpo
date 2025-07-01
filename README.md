
---

# SB3-GRPO: Group Relative Policy Optimization for Stable Baselines3

[[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Language Versions / 言語選択

- **English**: [README.md](README.md) (this file)
- **日本語**: [README_ja.md](README_ja.md)

---

`sb3-grpo` is a [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) (SB3) compatible implementation of **Group Relative Policy Optimization (GRPO)**.

This algorithm can be used as a drop-in replacement for standard PPO, providing stable learning especially in environments where rewards can be densely defined for states and actions.

## What is GRPO?

GRPO is an approach to the "credit assignment problem" in reinforcement learning.

### The Problem with Traditional PPO

Standard reinforcement learning algorithms like PPO face a fundamental challenge: **how to assign credit to actions based on future outcomes**.

PPO uses **GAE (Generalized Advantage Estimation)** to solve this:
1. **Collect episode data** with rewards received over time
2. **Estimate advantages** by looking at future rewards and value predictions
3. **Learn from temporal differences** between expected and actual outcomes

While effective, this approach has limitations:
- **Temporal delay**: Current actions are judged by distant future results
- **Estimation errors**: Value function inaccuracies propagate through GAE
- **Credit assignment ambiguity**: Hard to determine which specific action caused success/failure

### GRPO's Solution

GRPO takes a fundamentally different approach:

**"Instead of waiting for future outcomes, let's evaluate all possible actions right now and learn from their immediate relative quality."**

This is similar to multiple-choice questions:
*   **PPO with GAE**: Choose action A, wait for episode to end, then estimate "was A good?" based on final cumulative reward
*   **GRPO**: Choose action A while simultaneously computing "how good would actions B, C, D have been?" and learn "A was 1.2x better than B, 0.8x compared to C"

### Key Advantages

- **Immediate feedback**: No waiting for episode completion
- **Direct comparison**: Clear relative ranking of actions
- **Reduced variance**: Less dependency on value function estimation
- **Simplified learning**: No complex temporal credit assignment

## Installation

### Method 1: Direct installation from GitHub (Recommended)
```bash
pip install git+https://github.com/kechirojp/sb3-grpo.git
```

### Method 2: Development installation
```bash
git clone https://github.com/kechirojp/sb3-grpo.git
cd sb3-grpo
pip install -e .
```

### Method 3: Using requirements.txt
```bash
git clone https://github.com/kechirojp/sb3-grpo.git
cd sb3-grpo
pip install -r requirements.txt
```

**requirements.txt**
```txt
gymnasium>=0.26.0
torch>=1.11.0
stable-baselines3>=2.0.0
numpy>=1.20.0
```

## Usage: Complete Example (`CartPole-v1`)

Here's a complete sample code for training `CartPole-v1` environment using `GRPO`.

```python
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
```

### How to Run

Execute the following command in terminal:

```bash
python example.py
```

As training progresses, standard SB3 logs will be displayed. If the agent can maintain CartPole upright for extended periods after training, it's successful.

## API Reference

### GRPO Class

```python
class GRPO(PPO):
    """
    Group Relative Policy Optimization (GRPO) implementation extending PPO.
    
    Args:
        policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        env: The environment to learn from
        reward_function: Function to calculate rewards from (state, action, next_state)
        **kwargs: Other standard PPO arguments (learning_rate, n_steps, etc.)
    """
```

### Reward Function Interface

Your reward function must follow this signature:

```python
def your_reward_function(
    state: torch.Tensor,      # Current state [batch_size, state_dim]
    action: torch.Tensor,     # Action taken [batch_size, 1]  
    next_state: torch.Tensor  # Resulting state [batch_size, state_dim]
) -> torch.Tensor:            # Returns: rewards [batch_size, 1]
    # Your reward calculation logic here
    return rewards
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/kechirojp/sb3-grpo.git
cd sb3-grpo
pip install -e .[dev]  # Install with development dependencies
```

## License

This project is released under the MIT License. See the `LICENSE` file for details.

---
