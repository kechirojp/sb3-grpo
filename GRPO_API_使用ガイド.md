# GRPO API 使用ガイド

GRPOアルゴリズムをWebAPI経由で利用するための完全ガイドです。

## 🚀 クイックスタート

### 1. セットアップ

```bash
# 依存関係をインストール
pip install -r requirements.txt

# APIサーバーを起動
python run_api.py
```

サーバーは `http://localhost:8000` で起動します。

### 2. ブラウザでAPIドキュメントを確認

- **SwaggerUI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📋 基本的な使い方

### ステップ1: ヘルスチェック

まずAPIサーバーが正常に動作していることを確認します。

```python
import requests

# ヘルスチェック
response = requests.get("http://localhost:8000/health")
print(response.json())
# 出力例: {'status': 'healthy', 'timestamp': '2025-07-28T20:00:00', 'active_jobs': 0, 'loaded_models': 0}
```

### ステップ2: モデルの学習

CartPole環境でGRPOモデルを学習させます。

```python
# 学習設定
training_config = {
    "env_name": "CartPole-v1",           # 環境名
    "total_timesteps": 5000,             # 学習ステップ数
    "learning_rate": 3e-4,               # 学習率
    "n_steps": 32,                       # 環境ごとのステップ数
    "batch_size": 64,                    # バッチサイズ
    "n_epochs": 10,                      # エポック数
    "reward_function_type": "optimized_cartpole",  # 報酬関数
    "model_name": "my_cartpole_model"    # モデル名
}

# 学習開始
response = requests.post("http://localhost:8000/train", json=training_config)
result = response.json()
job_id = result["job_id"]
print(f"学習開始: Job ID = {job_id}")
```

### ステップ3: 学習進捗の監視

学習が完了するまで進捗を監視します。

```python
import time

# 学習進捗を監視
while True:
    response = requests.get(f"http://localhost:8000/training/{job_id}/status")
    status = response.json()
    
    progress = status['progress']
    current_steps = status['current_timesteps']
    total_steps = status['total_timesteps']
    job_status = status['status']
    
    print(f"進捗: {progress:.1%} ({current_steps}/{total_steps}) - 状態: {job_status}")
    
    if job_status in ["completed", "failed", "stopped"]:
        break
    
    time.sleep(2)  # 2秒待機

print(f"学習完了: 最終状態 = {job_status}")
```

### ステップ4: 学習済みモデルで推論

学習が完了したモデルを使って行動を予測します。

```python
# CartPoleの観測値 [cart_position, cart_velocity, pole_angle, pole_velocity]
observation = [0.1, 0.0, 0.05, 0.0]

# 推論実行
inference_request = {
    "model_name": "my_cartpole_model",
    "observation": observation,
    "deterministic": True  # 決定論的な行動選択
}

response = requests.post("http://localhost:8000/inference", json=inference_request)
result = response.json()

print(f"予測された行動: {result['action']}")  # 0 (左) または 1 (右)
if result.get('action_probs'):
    print(f"行動確率: {result['action_probs']}")
if result.get('value'):
    print(f"状態価値: {result['value']:.3f}")
```

## 🔧 高度な使用方法

### 複数モデルの管理

```python
# 利用可能なモデル一覧を取得
response = requests.get("http://localhost:8000/models")
models = response.json()

print("利用可能なモデル:")
for model in models["models"]:
    print(f"  - {model['name']} ({model['status']})")

# 不要なモデルを削除
model_to_delete = "old_model"
response = requests.delete(f"http://localhost:8000/models/{model_to_delete}")
if response.status_code == 200:
    print(f"モデル '{model_to_delete}' を削除しました")
```

### 複数の学習ジョブの実行

```python
# 複数の学習ジョブを並行実行
job_ids = []

for i in range(3):
    config = {
        "env_name": "CartPole-v1",
        "total_timesteps": 2000,
        "model_name": f"model_{i}",
        "learning_rate": 3e-4 * (i + 1)  # 異なる学習率
    }
    
    response = requests.post("http://localhost:8000/train", json=config)
    job_id = response.json()["job_id"]
    job_ids.append(job_id)
    print(f"ジョブ {i+1} 開始: {job_id}")

# すべてのジョブの状況を確認
response = requests.get("http://localhost:8000/training/jobs")
jobs = response.json()

print(f"\n実行中のジョブ数: {len([j for j in jobs['jobs'] if j['status'] == 'running'])}")
```

### カスタム報酬関数の使用

現在利用可能な報酬関数:

1. **optimized_cartpole**: CartPole用の最適化された報酬関数
2. **simple_cartpole**: シンプルな角度ベースの報酬関数

```python
# シンプルな報酬関数を使用
config = {
    "env_name": "CartPole-v1",
    "total_timesteps": 3000,
    "reward_function_type": "simple_cartpole",  # シンプル版を使用
    "model_name": "simple_model"
}

response = requests.post("http://localhost:8000/train", json=config)
```

## 🛠️ 実用的なサンプルコード

### 完全な学習・推論パイプライン

```python
import requests
import time
import numpy as np

class GRPOClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def train_model(self, model_name, env_name="CartPole-v1", timesteps=5000):
        """モデルを学習する"""
        config = {
            "env_name": env_name,
            "total_timesteps": timesteps,
            "model_name": model_name,
            "reward_function_type": "optimized_cartpole"
        }
        
        # 学習開始
        response = requests.post(f"{self.base_url}/train", json=config)
        job_id = response.json()["job_id"]
        
        print(f"🚀 学習開始: {model_name} (Job ID: {job_id})")
        
        # 学習完了まで待機
        while True:
            response = requests.get(f"{self.base_url}/training/{job_id}/status")
            status = response.json()
            
            if status["status"] == "completed":
                print(f"✅ 学習完了: {model_name}")
                return True
            elif status["status"] == "failed":
                print(f"❌ 学習失敗: {status.get('error_message', 'Unknown error')}")
                return False
            
            print(f"📈 進捗: {status['progress']:.1%}")
            time.sleep(3)
    
    def predict(self, model_name, observation):
        """推論を実行する"""
        data = {
            "model_name": model_name,
            "observation": observation,
            "deterministic": True
        }
        
        response = requests.post(f"{self.base_url}/inference", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"推論エラー: {response.text}")

# 使用例
client = GRPOClient()

# モデルを学習
if client.train_model("my_expert_model", timesteps=10000):
    # 学習済みモデルで推論
    test_observations = [
        [0.0, 0.0, 0.0, 0.0],      # 中央、静止
        [0.5, 0.1, 0.1, 0.2],     # 右寄り、右に傾く
        [-0.3, -0.1, -0.05, -0.1] # 左寄り、左に傾く
    ]
    
    for i, obs in enumerate(test_observations):
        result = client.predict("my_expert_model", obs)
        action_name = "右" if result["action"] == 1 else "左"
        print(f"観測 {i+1}: {obs} → 行動: {action_name} ({result['action']})")
```

### バッチ推論

```python
def batch_predict(model_name, observations):
    """複数の観測値に対して一度に推論を実行"""
    predictions = []
    
    for obs in observations:
        response = requests.post("http://localhost:8000/inference", json={
            "model_name": model_name,
            "observation": obs
        })
        
        if response.status_code == 200:
            predictions.append(response.json()["action"])
        else:
            predictions.append(None)
    
    return predictions

# 使用例
test_data = [
    [0.0, 0.0, 0.1, 0.0],
    [0.0, 0.0, -0.1, 0.0],
    [0.2, 0.0, 0.0, 0.0],
    [-0.2, 0.0, 0.0, 0.0]
]

actions = batch_predict("my_expert_model", test_data)
print("バッチ推論結果:", actions)
```

## 🔍 トラブルシューティング

### よくある問題と解決方法

#### 1. サーバーに接続できない

```python
import requests

try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print("✅ サーバー接続成功")
except requests.exceptions.ConnectionError:
    print("❌ サーバーに接続できません")
    print("解決方法:")
    print("1. 'python run_api.py' でサーバーを起動してください")
    print("2. ポート8000が他のプロセスで使用されていないか確認してください")
```

#### 2. 推論でモデルが見つからない

```python
# 利用可能なモデルを確認
response = requests.get("http://localhost:8000/models")
models = [m["name"] for m in response.json()["models"]]
print("利用可能なモデル:", models)

# 存在しないモデル名を指定していないか確認
```

#### 3. 学習が失敗する

```python
# 学習ジョブの詳細なエラー情報を確認
response = requests.get(f"http://localhost:8000/training/{job_id}/status")
status = response.json()

if status["status"] == "failed":
    print(f"エラー詳細: {status['error_message']}")
```

## 📊 パフォーマンス最適化

### 学習の高速化

```python
# より効率的な学習設定
fast_config = {
    "env_name": "CartPole-v1",
    "total_timesteps": 5000,
    "n_steps": 64,        # より大きなバッチ
    "batch_size": 128,    # バッチサイズを増加
    "n_epochs": 5,        # エポック数を削減
    "learning_rate": 1e-3 # 学習率を上げる
}
```

### メモリ使用量の最適化

```python
# 学習完了後、不要なモデルを削除
def cleanup_old_models():
    response = requests.get("http://localhost:8000/models")
    models = response.json()["models"]
    
    for model in models:
        if model["status"] == "saved":  # メモリに読み込まれていないモデル
            # 必要に応じて削除
            # requests.delete(f"http://localhost:8000/models/{model['name']}")
            pass
```

## 🌟 まとめ

このガイドでGRPO APIの基本的な使用方法から高度な活用方法まで学習できました：

1. **基本操作**: 学習 → 監視 → 推論のワークフロー
2. **モデル管理**: 複数モデルの同時管理
3. **実用的なクライアント**: 再利用可能なクライアントクラス
4. **トラブルシューティング**: よくある問題の解決方法

GRPO APIを使って効率的な強化学習システムを構築してください！

## 📞 サポート

- **APIドキュメント**: <http://localhost:8000/docs>
- **技術詳細**: `API_DOCUMENTATION.md`
- **実装詳細**: `grpo.py`、`api.py`
