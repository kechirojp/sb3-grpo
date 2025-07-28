# GRPO API ドキュメント

## 概要

このドキュメントは、GRPO（Group Relative Policy Optimization）強化学習アルゴリズム用のFastAPIサーバーについて説明します。APIは、モデルの学習、進捗監視、推論実行のためのエンドポイントを提供します。

**注意**: このWeb APIは補完的な機能です。このプロジェクトの核心的価値は、Stable Baselines3用のGRPOアルゴリズム実装にあります。

## クイックスタート

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. APIサーバーの起動

```bash
python run_api.py
```

サーバーは `http://localhost:8000` で起動します。

### 3. インタラクティブAPIドキュメントへのアクセス

ブラウザで以下にアクセス：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## APIエンドポイント

### ルート

#### `GET /`
基本的なAPI情報と利用可能なエンドポイントを取得

**レスポンス:**
```json
{
    "message": "GRPO API Server",
    "version": "1.0.0",
    "endpoints": {
        "train": "POST /train",
        "inference": "POST /inference", 
        "health": "GET /health",
        "models": "GET /models"
    }
}
```

### 学習

#### `POST /train`
新しい学習ジョブを開始

**リクエストボディ:**
```json
{
    "env_name": "CartPole-v1",
    "total_timesteps": 10000,
    "learning_rate": 0.0003,
    "n_steps": 32,
    "batch_size": 64,
    "n_epochs": 10,
    "reward_function_type": "optimized_cartpole",
    "model_name": "my_model"
}
```

**レスポンス:**
```json
{
    "job_id": "uuid-string",
    "message": "Training started",
    "config": {...}
}
```

#### `GET /training/{job_id}/status`
学習ジョブのステータスを取得

**レスポンス:**
```json
{
    "job_id": "uuid-string",
    "status": "running",
    "progress": 0.75,
    "current_timesteps": 7500,
    "total_timesteps": 10000,
    "start_time": "2025-07-28T10:00:00",
    "end_time": null,
    "error_message": null,
    "metrics": {}
}
```

#### `GET /training/jobs`
すべての学習ジョブを一覧表示

**レスポンス:**
```json
{
    "jobs": [
        {
            "job_id": "uuid-1",
            "status": "completed",
            "progress": 1.0
        },
        {
            "job_id": "uuid-2", 
            "status": "running",
            "progress": 0.3
        }
    ]
}
```

#### `POST /training/{job_id}/stop`
実行中の学習ジョブを停止

**レスポンス:**
```json
{
    "message": "Training job stopped",
    "job_id": "uuid-string"
}
```

### 推論

#### `POST /inference`
学習済みモデルを使用して予測を実行

**リクエストボディ:**
```json
{
    "model_name": "my_model",
    "observation": [0.1, 0.0, 0.05, 0.0],
    "deterministic": true
}
```

**レスポンス:**
```json
{
    "action": 1,
    "action_probs": [0.3, 0.7],
    "value": 0.85
}
```

**注意**: 
- `observation`: 単一の観測値配列（CartPoleの場合は4つの値）
- `action`: 単一の予測行動（整数）
- `action_probs`: すべての可能な行動の確率分布（オプション）
- `value`: 推定状態価値（オプション）

### モデル管理

#### `GET /models`
利用可能なすべてのモデルを一覧表示

**レスポンス:**
```json
{
    "models": [
        {"name": "my_model", "status": "loaded"},
        {"name": "old_model", "status": "saved"}
    ]
}
```

#### `DELETE /models/{model_name}`
メモリとディスクからモデルを削除

### ヘルスチェック

#### `GET /health`
APIサーバーのヘルス状態をチェック

**レスポンス:**
```json
{
    "status": "healthy",
    "timestamp": "2025-07-28T10:00:00",
    "active_jobs": 1,
    "loaded_models": 2
}
```

## 設定

### 学習パラメータ

すべてのパラメータはオプションで、適切なデフォルト値があります：

- `env_name`: Gymnasium環境名（デフォルト: "CartPole-v1"）
- `total_timesteps`: 総学習ステップ数（デフォルト: 10000）
- `learning_rate`: 学習率（デフォルト: 3e-4）
- `n_steps`: 環境ごとのステップ数（デフォルト: 32）
- `batch_size`: バッチサイズ（デフォルト: 64）
- `n_epochs`: 学習エポック数（デフォルト: 10）
- `reward_function_type`: 報酬関数のタイプ（デフォルト: "optimized_cartpole"）
- `model_name`: モデルを保存する名前（オプション、指定されない場合は自動生成）

### 利用可能な報酬関数

1. **optimized_cartpole**: CartPole用の指数報酬で性能最適化
2. **simple_cartpole**: CartPole用のシンプルな角度ベース報酬

### モデル保存

- モデルは自動的に `models/` ディレクトリに保存されます
- モデルファイルはZIP形式で圧縮されます
- 高速推論のため、初回ロード後はモデルがメモリにキャッシュされます

## 使用例

### Pythonクライアント

```python
import requests

# 学習開始
response = requests.post("http://localhost:8000/train", json={
    "env_name": "CartPole-v1",
    "total_timesteps": 5000,
    "model_name": "test_model"
})
job_id = response.json()["job_id"]

# ステータス確認
status = requests.get(f"http://localhost:8000/training/{job_id}/status")
print(status.json())

# 予測実行
prediction = requests.post("http://localhost:8000/inference", json={
    "model_name": "test_model",
    "observation": [0.1, 0.0, 0.05, 0.0]
})
print(prediction.json())
```

### cURLの例

```bash
# 学習開始
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"env_name": "CartPole-v1", "total_timesteps": 5000}'

# ヘルスチェック
curl "http://localhost:8000/health"

# モデル一覧
curl "http://localhost:8000/models"
```

## プロジェクト構造

```text
├── models/                    # 保存されたモデルファイル（自動作成）
│   ├── model_1.zip
│   └── model_2.zip
├── api.py                     # メインのFastAPIアプリケーション
├── run_api.py                 # サーバー起動スクリプト
├── grpo.py                    # GRPOアルゴリズム実装
├── requirements.txt           # 依存関係
└── README.md                  # プロジェクトドキュメント
```

## エラーハンドリング

APIは標準的なHTTPステータスコードを返します：

- `200`: 成功
- `400`: 不正なリクエスト（無効なパラメータ）
- `404`: リソースが見つからない（ジョブ/モデル）
- `422`: バリデーションエラー（不正な形式のリクエスト）
- `500`: 内部サーバーエラー

エラーレスポンスには詳細が含まれます：
```json
{
    "detail": "何が問題だったかを説明するエラーメッセージ"
}
```

## パフォーマンスに関する注意事項

- **モデルキャッシュ**: 高速推論のため、初回ロード後はモデルがメモリにキャッシュされます
- **バックグラウンド学習**: 学習はAPIをブロックしないバックグラウンドスレッドで実行されます
- **並行処理**: 複数の学習ジョブと推論リクエストを同時実行可能
- **GPU対応**: 利用可能な場合は自動的にCUDAを使用し、CPUにフォールバック
- **メモリ管理**: メモリ不足時にはモデルが自動的にアンロードされます

## 開発

### カスタム報酬関数の追加

`api.py`の`REWARD_FUNCTIONS`辞書に新しい報酬関数を追加：

```python
def my_custom_reward(state, action, next_state):
    # ここに報酬ロジックを記述
    return reward_tensor

REWARD_FUNCTIONS["my_custom"] = my_custom_reward
```

### 新しい環境への拡張

1. 環境固有の報酬関数を追加
2. 学習設定のバリデーションを更新
3. 新しい環境でテスト
4. ドキュメントを更新

### API開発

APIはFastAPIで構築され、以下を含みます：
- 自動OpenAPIドキュメント生成
- Pydanticによる入力バリデーション
- Webアプリケーション用のCORSサポート
- バックグラウンドタスク処理

### セキュリティに関する考慮事項

**重要**: このAPIは開発・研究環境向けに設計されています。本番展開では以下を検討してください：

1. **認証/承認**: APIキーまたはOAuth統合の追加
2. **レート制限**: 悪用防止のためのリクエストレート制限の実装
3. **入力バリデーション**: モデル名やファイルパスの追加バリデーション
4. **HTTPS設定**: 暗号化通信のためのSSL/TLSの使用
5. **リソース制限**: 学習ジョブ実行時間とモデルサイズの制限設定
6. **ネットワークセキュリティ**: アクセス制限のためのファイアウォールとVPNの使用
7. **監視**: セキュリティイベントのログ記録と監視の追加

### サポート

問題や機能リクエストについては：

- **GitHub Issues**: [sb3-grpo repository](https://github.com/kechirojp/sb3-grpo/issues)
- **ドキュメント**: GRPOアルゴリズムの詳細については、メインの[README.md](README.md)を参照

---

*注意: このWeb APIは補完的な機能です。このプロジェクトの核心的価値は、Stable Baselines3用のGRPOアルゴリズム実装にあります。*
