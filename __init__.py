# sb3-grpo: Group Relative Policy Optimization for Stable Baselines3

"""
sb3-grpo パッケージ

このパッケージは、Stable Baselines3と互換性のある
Group Relative Policy Optimization (GRPO) の実装を提供します。

Usage:
    from sb3_grpo import GRPO
    
    # 報酬関数を定義
    def my_reward_function(state, action, next_state):
        # あなたの報酬計算ロジック
        return reward
    
    # エージェントを作成
    agent = GRPO(
        "MlpPolicy",
        env,
        reward_function=my_reward_function,
        verbose=1
    )
    
    # 学習実行
    agent.learn(total_timesteps=10000)
"""

from .grpo import GRPO

__version__ = "0.1.0"
__author__ = "kechirojp"
__email__ = "kechirojp@gmail.com"

# パッケージの公開API
__all__ = [
    "GRPO",
]
