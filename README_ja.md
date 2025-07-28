
---

# SB3-GRPO: Group Relative Policy Optimization for Stable Baselines3

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.11%2B-orange.svg)
![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0%2B-red.svg)

## Language Versions / 言語選択

- **English**: [README.md](README.md)
- **日本語**: [README_ja.md](README_ja.md) (このファイル)

---

`sb3-grpo` は、[Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) (SB3) と互換性のある、**Group Relative Policy Optimization (GRPO)** の実装です。

このアルゴリズムは、標準的なPPOを置き換える形で利用でき、特に状態と行動に対する報酬が密に定義できる環境で、安定した学習を提供することを目指します。

## GRPOとは？

GRPOは、強化学習における「信用割り当て問題」に対する一つのアプローチです。

### 従来のPPOの課題

PPOなどの標準的な強化学習アルゴリズムは、根本的な課題に直面しています：**将来の結果に基づいて行動にどう信用を割り当てるか**です。

PPOは**GAE (Generalized Advantage Estimation)**でこれを解決します：
1. **エピソードデータを収集**し、時間経過とともに報酬を記録
2. **将来の報酬と価値予測を見て**アドバンテージを推定
3. **期待値と実際の結果の時間的差分**から学習

効果的ではありますが、この手法には限界があります：
- **時間的遅延**: 現在の行動が遠い未来の結果で判断される
- **推定誤差**: 価値関数の不正確さがGAE全体に伝播
- **信用割り当ての曖昧さ**: どの具体的な行動が成功/失敗を引き起こしたか判定困難

### GRPOの解決策

GRPOは根本的に異なるアプローチを取ります：

**「将来の結果を待つのではなく、今この瞬間に全ての可能なアクションを評価し、それらの即座の相対的品質から学習しよう」**

これは、多肢選択問題に似ています：
*   **PPO + GAE**: アクションAを選択、エピソード終了まで待機、最終的な累積報酬で「Aは良かったか？」を推定
*   **GRPO**: アクションAを選択すると同時に「アクションB、C、Dはどの程度良かったか？」を計算し、「AはBより1.2倍良く、Cと比べて0.8倍」という学習

### 主な利点

- **即座のフィードバック**: エピソード完了を待つ必要なし
- **直接比較**: アクションの明確な相対ランキング
- **分散の削減**: 価値関数推定への依存度が低い
- **学習の簡素化**: 複雑な時間的信用割り当てが不要

## インストール

### 方法1: GitHubから直接インストール（推奨）
```bash
pip install git+https://github.com/kechirojp/sb3-grpo.git
```

### 方法2: 開発用インストール
```bash
git clone https://github.com/kechirojp/sb3-grpo.git
cd sb3-grpo
pip install -e .
```

### 方法3: requirements.txtを使用
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

## 使い方：完全な実行例 (`CartPole-v1`)

以下は、`GRPO` を使って `CartPole-v1` 環境を学習させるための完全なサンプルコードです。

```python
# example.py

import gymnasium as gym
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

# パッケージからGRPOをインポート
from sb3_grpo import GRPO

# --- 1. GRPO用の報酬関数を定義 ---
# GRPOの核心は、この報酬関数を外部から注入できる点にあります。
# ここでは、CartPoleの「次の状態」を見て、どれだけ良い状態かを評価する関数を定義します。
def cartpole_reward_fn(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    """
    CartPole環境用の報酬関数。
    次の状態 (next_state) がどれだけ「良い」かを評価して報酬を返す。
    - ポールの角度が垂直に近いほど高報酬
    - カートが中央に近いほど高報酬
    """
    # next_state の中身: [カート位置, カート速度, ポール角度, ポール角速度]
    cart_pos = next_state[:, 0]
    pole_angle = next_state[:, 2]
    
    # 角度が0に近いほど、位置が0に近いほど報酬が高くなるように設計
    reward = 1.0 - torch.abs(pole_angle) - 0.1 * torch.abs(cart_pos)
    
    return reward.unsqueeze(-1)


# --- 2. 環境のセットアップ ---
# Stable Baselines3の標準的な方法で環境を準備
env = gym.make("CartPole-v1")
env = DummyVecEnv([lambda: env])


# --- 3. GRPOエージェントの作成 ---
# PPOをインスタンス化するのとほぼ同じ使い方です。
agent = GRPO(
    "MlpPolicy",
    env,
    reward_function=cartpole_reward_fn,  # ここで報酬関数を注入
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    verbose=1,
)

# --- 4. 学習の実行 ---
# あとはSB3のPPOと全く同じように `learn` メソッドを呼び出すだけ
print("--- Starting GRPO Training ---")
agent.learn(total_timesteps=20000)
print("--- Training Finished ---")


# --- 5. 学習済みエージェントの評価 ---
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

### 実行方法

ターミナルで以下のコマンドを実行します。

```bash
python example.py
```

学習が進むにつれて、SB3の標準的なログが出力されます。学習完了後、エージェントがCartPoleを長時間倒さずに維持できれば成功です。

## APIリファレンス

### GRPOクラス

```python
class GRPO(PPO):
    """
    Group Relative Policy Optimization (GRPO) PPOを拡張した実装
    
    Args:
        policy: 使用するポリシーモデル (MlpPolicy, CnnPolicy, ...)
        env: 学習する環境
        reward_function: (state, action, next_state)から報酬を計算する関数
        **kwargs: その他の標準PPO引数 (learning_rate, n_steps, など)
    """
```

### 報酬関数インターフェース

報酬関数は以下のシグネチャに従う必要があります：

```python
def your_reward_function(
    state: torch.Tensor,      # 現在の状態 [batch_size, state_dim]
    action: torch.Tensor,     # 実行された行動 [batch_size, 1]  
    next_state: torch.Tensor  # 結果の状態 [batch_size, state_dim]
) -> torch.Tensor:            # 返り値: 報酬 [batch_size, 1]
    # ここに報酬計算ロジックを記述
    return rewards
```

## 🚀 追加機能: WebAPI

**オプション**: GRPOはリモートサーバー展開用のWebAPIとしても利用可能です：

```bash
# APIサーバー起動
python run_api.py
# サーバーが利用可能になります: http://your-server:8000
```

```python
# モデル学習
import requests
response = requests.post("http://your-server:8000/train", json={
    "env_name": "CartPole-v1", "total_timesteps": 1000, "model_name": "test"
})

# 推論実行
prediction = requests.post("http://your-server:8000/inference", json={
    "model_name": "test", "observation": [0.1, 0.0, 0.05, 0.0]
})
print(f"行動: {prediction.json()['action']}")
```

📖 **APIドキュメント**:

- **入門ガイド**: [はじめに.md](はじめに.md)
- **完全ガイド**: [GRPO_API_使用ガイド.md](GRPO_API_使用ガイド.md)
- **API仕様**: [API_DOCUMENTATION_ja.md](API_DOCUMENTATION_ja.md)
- **インタラクティブドキュメント**: <http://localhost:8000/docs>

*注意: WebAPIは補完的な機能です。このプロジェクトの核心的価値は、Stable Baselines3用のGRPOアルゴリズム実装にあります。*

## コントリビューション

コントリビューションを歓迎します！プルリクエストをお気軽に提出してください。大きな変更については、まずissueを開いて議論することをお勧めします。

### 開発環境のセットアップ

```bash
git clone https://github.com/kechirojp/sb3-grpo.git
cd sb3-grpo
pip install -e .[dev]  # 開発用依存関係と一緒にインストール
```

## 言語バージョン

- **English**: [README.md](README.md)
- **日本語**: [README_ja.md](README_ja.md) (このファイル)

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は `LICENSE` ファイルをご覧ください。

---