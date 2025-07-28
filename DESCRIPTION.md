# SB3-GRPO: Group Relative Policy Optimization for Stable Baselines3

**🎯 簡潔な説明**
Group Relative Policy Optimization (GRPO) アルゴリズムをStable Baselines3環境で利用できるようにしたPythonライブラリです。

**🔧 主な特徴**
- PPOと同じAPIで簡単に置き換え可能
- pip installでワンコマンドインストール
- CartPole-v1での動作確認済み
- 日英両言語でのドキュメント提供

**📈 技術的価値**
従来のPPOでは行動の良し悪しを判断するのに時間がかかりましたが、GRPOは行動を取った瞬間に複数の選択肢を比較評価できるため、より効率的な学習が可能です。

**🚀 使い方**
```bash
pip install git+https://github.com/kechirojp/sb3-grpo.git
```

```python
from sb3_grpo import GRPO
agent = GRPO("MlpPolicy", env, reward_function=custom_reward_fn)
agent.learn(total_timesteps=20000)
```

**📊 実用例**
DeepSeekが大規模言語モデルの学習でGRPOアプローチを採用するなど、商用レベルでの有効性が実証されています。

**🎓 学習価値**
論文で提案されたアルゴリズムを実際に使えるライブラリとして実装することで、強化学習の理論と実装の橋渡しを実現しています。
