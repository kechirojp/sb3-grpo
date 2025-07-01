# grpo.py

from typing import Callable, Optional, Tuple, Any
import warnings

import numpy as np
import torch
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

class GRPO(PPO):
    """
    Group Relative Policy Optimization (GRPO) implementation extending PPO.

    This algorithm evaluates all possible actions as a "group" at each step,
    directly calculating the relative goodness of rewards within that group as advantages.
    This eliminates the need for GAE (Generalized Advantage Estimation).

    Args:
        policy: The policy model to use (MlpPolicy, CnnPolicy, ...).
        env: The environment to learn from (if registered in Gym, can be str).
        reward_function (Callable): Function to calculate rewards from state, action, and next state.
            Signature: (state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor
            This function serves as the source of GRPO rewards and can be injected externally.
        **kwargs: Other standard PPO arguments (learning_rate, n_steps, batch_size, etc.).
    """
    def __init__(
        self,
        policy: Any,
        env: VecEnv,
        reward_function: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        **kwargs,
    ):
        # GRPO needs to evaluate all actions at each step, so buffer size must be multiplied by action count
        # If user specifies n_steps, adjust it internally for GRPO
        if "n_steps" in kwargs:
            original_n_steps = kwargs["n_steps"]
            # Actual buffer steps = original n_steps * number of actions
            kwargs["n_steps"] = original_n_steps * env.action_space.n
            self.original_n_steps = original_n_steps
        else:
            # Use default value
            self.original_n_steps = PPO.n_steps
            kwargs["n_steps"] = self.original_n_steps * env.action_space.n
        
        # Initialize parent class (PPO)
        super().__init__(policy=policy, env=env, **kwargs)
        
        # Store externally injected reward function
        self.reward_function = reward_function

        # Error if action space is not Discrete
        if not isinstance(self.action_space, spaces.Discrete):
            raise NotImplementedError("GRPO currently only supports Discrete action spaces.")

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect rollouts (experience data) using GRPO logic.
        This method completely overrides PPO's `collect_rollouts`.

        Args:
            env: The vectorized environment
            callback: The callback to use
            rollout_buffer: The buffer to collect rollouts
            n_rollout_steps: The number of steps to collect (adjusted for GRPO)

        Returns:
            True if training should continue, False otherwise.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Reset the buffer
        rollout_buffer.reset()
        # Set policy to evaluation mode
        self.policy.set_training_mode(False)

        # Initialize callback
        callback.on_rollout_start()

        # --- Core GRPO logic starts here ---
        
        # Interact with environment for original_n_steps times
        for _ in range(self.original_n_steps):
            # Per-step callback processing
            if callback.on_step() is False:
                return False

            # 1. Sample one action to actually advance the environment
            with torch.no_grad():
                # Get action using SB3's standard method
                actions, values, log_probs = self.policy(torch.as_tensor(self._last_obs).to(self.device))
            
            # 2. Advance environment one step with that action and get next state
            new_obs, rewards, dones, infos = env.step(actions.cpu().numpy())
            self.num_timesteps += env.num_envs

            # 3. Define evaluation group (all actions)
            num_actions = self.action_space.n
            all_actions = np.arange(num_actions)

            # 4. Batch calculate rewards and values for all actions
            with torch.no_grad():
                # Convert data to tensors and send to device for computation
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device)
                new_obs_tensor = torch.as_tensor(new_obs).to(self.device)
                all_actions_tensor = torch.as_tensor(all_actions, device=self.device)

                # Replicate data for batch computation
                # (num_envs, obs_dim) -> (num_envs * num_actions, obs_dim)
                # Example: [[env1_obs], [env2_obs]] -> [[env1_obs], [env1_obs], ..., [env2_obs], [env2_obs], ...]
                repeated_obs = obs_tensor.repeat_interleave(num_actions, dim=0)
                repeated_new_obs = new_obs_tensor.repeat_interleave(num_actions, dim=0)
                
                # (num_actions) -> (num_envs * num_actions)
                # Example: [0, 1, 2] -> [0, 1, 2, 0, 1, 2, ...]
                tiled_actions = all_actions_tensor.repeat(env.num_envs)

                # *** Calculate reward source using externally injected function ***
                # This function takes (state, action, next_state)
                final_rewards = self.reward_function(
                    repeated_obs, tiled_actions.unsqueeze(-1), repeated_new_obs
                )
                
                # Batch calculate values and log probabilities for all actions
                group_values, group_log_probs, _ = self.policy.evaluate_actions(repeated_obs, tiled_actions)

            # 5. Calculate GRPO advantages (reward normalization)
            # (num_envs * num_actions, 1) -> (num_envs, num_actions)
            final_rewards_reshaped = final_rewards.reshape(env.num_envs, num_actions)
            
            # Calculate mean and standard deviation for each environment
            mean_reward = final_rewards_reshaped.mean(dim=1, keepdim=True)
            std_reward = final_rewards_reshaped.std(dim=1, keepdim=True)
            
            # Advantage = (individual reward - mean reward) / (std deviation + ε)
            advantages = (final_rewards_reshaped - mean_reward) / (std_reward + 1e-10)

            # 6. Store all action experiences in buffer
            # Buffer has flat structure, so reshape data before adding
            # (num_envs, num_actions) -> (num_envs * num_actions)
            flat_advantages = advantages.flatten()
            flat_group_values = group_values.flatten()
            flat_group_log_probs = group_log_probs.flatten()
            
            # Buffer's add method expects (num_envs, ...) shape,
            # so loop num_actions times to add data
            for i in range(num_actions):
                # Prepare i-th action for all environments
                action_slice = np.array([all_actions[i]] * env.num_envs)
                
                # Slice of advantages, values, and log probabilities corresponding to i-th action
                adv_slice = advantages[:, i].cpu().numpy()
                val_slice = group_values.reshape(env.num_envs, num_actions)[:, i]
                log_prob_slice = group_log_probs.reshape(env.num_envs, num_actions)[:, i]

                # Add to buffer. Key point: store advantages as rewards
                rollout_buffer.add(
                    self._last_obs, action_slice, adv_slice, self._last_episode_starts, val_slice, log_prob_slice
                )

            # 7. Update state
            self._last_obs = new_obs
            self._last_episode_starts = dones

        # --- End of core GRPO logic ---

        # Post-processing after rollout collection (SB3 standard)
        with torch.no_grad():
            # Calculate value of last state (needed for GAE computation)
            last_values, _, _ = self.policy(torch.as_tensor(new_obs).to(self.device))

        # Let buffer calculate returns and advantages
        # At this point, GAE is calculated based on rewards (GRPO advantages)
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
        
        # *** Final GRPO touch ***
        # Overwrite GAE-calculated advantages with our pure GRPO advantages.
        # This operation is essential as PPO's loss function references `rollout_buffer.advantages`.
        rollout_buffer.advantages = rollout_buffer.rewards.copy()

        callback.on_rollout_end()

        return True