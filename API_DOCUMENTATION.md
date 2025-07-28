# GRPO API Documentation

## Overview

This document describes the FastAPI server for the GRPO (Group Relative Policy Optimization) reinforcement learning algorithm. The API provides endpoints for training models, monitoring progress, and performing inference.

**Note**: This Web API is a supplementary feature. The core value of this project is the GRPO algorithm implementation for Stable Baselines3.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python run_api.py
```

The server will start at `http://localhost:8000`.

### 3. Access Interactive API Documentation

Open your browser and go to:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Root

#### `GET /`
Get basic API information and available endpoints.

**Response:**
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

### Training

#### `POST /train`
Start a new training job.

**Request Body:**
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

**Response:**
```json
{
    "job_id": "uuid-string",
    "message": "Training started",
    "config": {...}
}
```

#### `GET /training/{job_id}/status`
Get the status of a training job.

**Response:**
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
List all training jobs.

**Response:**
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
Stop a running training job.

**Response:**
```json
{
    "message": "Training job stopped",
    "job_id": "uuid-string"
}
```

### Inference

#### `POST /inference`
Make predictions using a trained model.

**Request Body:**
```json
{
    "model_name": "my_model",
    "observation": [0.1, 0.0, 0.05, 0.0],
    "deterministic": true
}
```

**Response:**
```json
{
    "action": 1,
    "action_probs": [0.3, 0.7],
    "value": 0.85
}
```

**Note**: 
- `observation`: Single observation array (4 values for CartPole)
- `action`: Single predicted action (integer)
- `action_probs`: Probability distribution over all possible actions (optional)
- `value`: Estimated state value (optional)
```

### Model Management

#### `GET /models`
List all available models.

**Response:**
```json
{
    "models": [
        {"name": "my_model", "status": "loaded"},
        {"name": "old_model", "status": "saved"}
    ]
}
```

#### `DELETE /models/{model_name}`
Delete a model from memory and disk.

### Health Check

#### `GET /health`
Check API server health.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-07-28T10:00:00",
    "active_jobs": 1,
    "loaded_models": 2
}
```

## Configuration

### Training Parameters

All parameters are optional with sensible defaults:

- `env_name`: Gymnasium environment name (default: "CartPole-v1")
- `total_timesteps`: Total training steps (default: 10000)
- `learning_rate`: Learning rate (default: 3e-4)
- `n_steps`: Steps per environment (default: 32)
- `batch_size`: Batch size (default: 64)
- `n_epochs`: Training epochs (default: 10)
- `reward_function_type`: Type of reward function (default: "optimized_cartpole")
- `model_name`: Name to save the model (optional, auto-generated if not provided)

### Available Reward Functions

1. **optimized_cartpole**: Performance-optimized for CartPole with exponential rewards
2. **simple_cartpole**: Simple angle-based reward for CartPole

### Model Storage

- Models are automatically saved to the `models/` directory
- Model files are compressed using ZIP format
- Models are cached in memory after first load for fast inference

## Usage Examples

### Python Client

```python
import requests

# Start training
response = requests.post("http://localhost:8000/train", json={
    "env_name": "CartPole-v1",
    "total_timesteps": 5000,
    "model_name": "test_model"
})
job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:8000/training/{job_id}/status")
print(status.json())

# Make prediction
prediction = requests.post("http://localhost:8000/inference", json={
    "model_name": "test_model",
    "observation": [0.1, 0.0, 0.05, 0.0]
})
print(prediction.json())
```

### cURL Examples

```bash
# Start training
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"env_name": "CartPole-v1", "total_timesteps": 5000}'

# Check health
curl "http://localhost:8000/health"

# List models
curl "http://localhost:8000/models"
```

## Project Structure

```text
├── models/                    # Saved model files (auto-created)
│   ├── model_1.zip
│   └── model_2.zip
├── api.py                     # Main FastAPI application
├── run_api.py                 # Server startup script
├── grpo.py                    # GRPO algorithm implementation
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Resource not found (job/model)
- `422`: Validation error (malformed request)
- `500`: Internal server error

Error responses include details:
```json
{
    "detail": "Error message describing what went wrong"
}
```

## Performance Notes

- **Model Caching**: Models are cached in memory after first load for fast inference
- **Background Training**: Training runs in background threads without blocking the API
- **Concurrent Operations**: Multiple training jobs and inference requests can run simultaneously
- **GPU Support**: Automatically uses CUDA if available, falls back to CPU
- **Memory Management**: Models are automatically unloaded when memory is low

## Development

### Adding Custom Reward Functions

Add new reward functions to the `REWARD_FUNCTIONS` dictionary in `api.py`:

```python
def my_custom_reward(state, action, next_state):
    # Your reward logic here
    return reward_tensor

REWARD_FUNCTIONS["my_custom"] = my_custom_reward
```

### Extending for New Environments

1. Add environment-specific reward functions
2. Update the training configuration validation
3. Test with the new environment
4. Update documentation

### API Development

The API is built with FastAPI and includes:
- Automatic OpenAPI documentation
- Input validation with Pydantic
- CORS support for web applications
- Background task processing

### Security Considerations

**Important**: This API is designed for development and research environments. For production deployment, consider:

1. **Authentication/Authorization**: Add API keys or OAuth integration
2. **Rate Limiting**: Implement request rate limiting to prevent abuse
3. **Input Validation**: Additional validation for model names and file paths
4. **HTTPS Configuration**: Use SSL/TLS for encrypted communication
5. **Resource Limits**: Set limits on training job duration and model sizes
6. **Network Security**: Use firewalls and VPNs to restrict access
7. **Monitoring**: Add logging and monitoring for security events

### Support

For issues and feature requests:
- **GitHub Issues**: [sb3-grpo repository](https://github.com/kechirojp/sb3-grpo/issues)
- **Documentation**: See the main [README.md](README.md) for GRPO algorithm details

---

*Note: This Web API is a supplementary feature. The core value of this project is the GRPO algorithm implementation for Stable Baselines3.*
