# api.py - FastAPI server for GRPO training and inference

import asyncio
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import gymnasium as gym
import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from grpo import GRPO

# --- Data Models ---

class TrainingConfig(BaseModel):
    env_name: str = Field(default="CartPole-v1", description="Gymnasium environment name")
    total_timesteps: int = Field(default=10000, description="Total training timesteps")
    learning_rate: float = Field(default=3e-4, description="Learning rate")
    n_steps: int = Field(default=32, description="Number of steps to run for each environment")
    batch_size: int = Field(default=64, description="Batch size")
    n_epochs: int = Field(default=10, description="Number of epochs when optimizing the surrogate loss")
    reward_function_type: str = Field(default="optimized_cartpole", description="Type of reward function")
    model_name: Optional[str] = Field(default=None, description="Name to save the model as")

class TrainingStatus(BaseModel):
    job_id: str
    status: str  # "running", "completed", "failed", "stopped"
    progress: float  # 0.0 to 1.0
    current_timesteps: int
    total_timesteps: int
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)

class InferenceRequest(BaseModel):
    model_name: str = Field(description="Name of the trained model")
    observation: List[float] = Field(description="Environment observation")
    deterministic: bool = Field(default=True, description="Use deterministic action selection")

class InferenceResponse(BaseModel):
    action: int
    action_probs: Optional[List[float]] = None
    value: Optional[float] = None

# --- Global state ---
training_jobs: Dict[str, TrainingStatus] = {}
running_models: Dict[str, GRPO] = {}
model_storage_path = "models"

# Ensure model storage directory exists
os.makedirs(model_storage_path, exist_ok=True)

# --- FastAPI app ---
app = FastAPI(
    title="GRPO API Server",
    description="API server for Group Relative Policy Optimization training and inference",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Reward Functions ---

def optimized_cartpole_reward_fn(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    """Performance-optimized reward function for CartPole."""
    cart_pos = next_state[:, 0]
    pole_angle = next_state[:, 2]
    
    angle_reward = 3.0 * torch.exp(-8.0 * torch.abs(pole_angle))
    position_reward = 1.0 * torch.exp(-1.5 * cart_pos**2)
    survival_bonus = 1.5
    
    boundary_penalty = torch.where(
        torch.abs(cart_pos) > 2.0,
        -2.0 * (torch.abs(cart_pos) - 2.0),
        torch.zeros_like(cart_pos)
    )
    
    total_reward = survival_bonus + angle_reward + position_reward + boundary_penalty
    total_reward = torch.clamp(total_reward, min=0.1, max=6.0)
    
    return total_reward.unsqueeze(-1)

def simple_cartpole_reward_fn(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    """Simple reward function for CartPole."""
    pole_angle = next_state[:, 2]
    reward = 1.0 - torch.abs(pole_angle) / 0.2095
    reward = torch.clamp(reward, min=0.1, max=1.0)
    return reward.unsqueeze(-1)

REWARD_FUNCTIONS = {
    "optimized_cartpole": optimized_cartpole_reward_fn,
    "simple_cartpole": simple_cartpole_reward_fn,
}

# --- Training callback ---

class APITrainingCallback(BaseCallback):
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id
        
    def _on_step(self) -> bool:
        if self.job_id in training_jobs:
            job = training_jobs[self.job_id]
            job.current_timesteps = self.num_timesteps
            job.progress = min(self.num_timesteps / job.total_timesteps, 1.0)
            
            # Update metrics if available
            if hasattr(self.locals, 'infos') and self.locals['infos']:
                for info in self.locals['infos']:
                    if isinstance(info, dict):
                        for key, value in info.items():
                            if isinstance(value, (int, float)):
                                job.metrics[key] = float(value)
        
        return True

# --- Training function ---

def train_model_background(job_id: str, config: TrainingConfig):
    """Background training function."""
    try:
        job = training_jobs[job_id]
        job.status = "running"
        
        # Setup environment
        env = gym.make(config.env_name)
        env = DummyVecEnv([lambda: env])
        
        # Get reward function
        if config.reward_function_type not in REWARD_FUNCTIONS:
            raise ValueError(f"Unknown reward function: {config.reward_function_type}")
        
        reward_fn = REWARD_FUNCTIONS[config.reward_function_type]
        
        # Create GRPO model
        model = GRPO(
            policy="MlpPolicy",
            env=env,
            reward_function=reward_fn,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            verbose=1
        )
        
        # Setup callback
        callback = APITrainingCallback(job_id)
        
        # Train model
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback
        )
        
        # Save model
        model_name = config.model_name or f"grpo_model_{job_id}"
        model_path = os.path.join(model_storage_path, f"{model_name}.zip")
        model.save(model_path)
        
        # Store model in memory for inference
        running_models[model_name] = model
        
        # Update job status
        job.status = "completed"
        job.end_time = datetime.now()
        job.progress = 1.0
        
    except Exception as e:
        job.status = "failed"
        job.end_time = datetime.now()
        job.error_message = str(e)

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "GRPO API Server",
        "version": "1.0.0",
        "endpoints": {
            "training": "/train",
            "status": "/training/{job_id}/status",
            "jobs": "/training/jobs",
            "inference": "/inference",
            "models": "/models"
        }
    }

@app.post("/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start a new training job."""
    job_id = str(uuid.uuid4())
    
    # Create training job
    job = TrainingStatus(
        job_id=job_id,
        status="starting",
        progress=0.0,
        current_timesteps=0,
        total_timesteps=config.total_timesteps,
        start_time=datetime.now()
    )
    
    training_jobs[job_id] = job
    
    # Start training in background
    background_tasks.add_task(train_model_background, job_id, config)
    
    return {"job_id": job_id, "message": "Training started"}

@app.get("/training/{job_id}/status")
async def get_training_status(job_id: str):
    """Get status of a training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return training_jobs[job_id]

@app.get("/training/jobs")
async def list_training_jobs():
    """List all training jobs."""
    return {"jobs": list(training_jobs.values())}

@app.post("/training/{job_id}/stop")
async def stop_training(job_id: str):
    """Stop a running training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    if job.status == "running":
        job.status = "stopped"
        job.end_time = datetime.now()
        return {"message": "Training stopped"}
    else:
        return {"message": f"Job is not running (status: {job.status})"}

@app.post("/inference")
async def predict(request: InferenceRequest):
    """Make prediction using a trained model."""
    if request.model_name not in running_models:
        # Try to load from disk
        model_path = os.path.join(model_storage_path, f"{request.model_name}.zip")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        try:
            # Create a simple environment for loading
            env = gym.make("CartPole-v1")
            env = DummyVecEnv([lambda: env])
            
            # Load the model using PPO.load and convert it to GRPO-like object
            from stable_baselines3 import PPO
            loaded_model = PPO.load(model_path, env=env)
            
            # Store the loaded model
            running_models[request.model_name] = loaded_model
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    model = running_models[request.model_name]
    
    try:
        # Convert observation to numpy array
        obs = np.array(request.observation, dtype=np.float32)
        
        # Get action
        action, _states = model.predict(obs, deterministic=request.deterministic)
        
        # Handle both scalar and array action outputs
        if isinstance(action, np.ndarray):
            if action.ndim == 0:  # scalar (0-dimensional array)
                action_int = int(action.item())
            else:  # array
                action_int = int(action[0])
        else:
            action_int = int(action)
        
        response = InferenceResponse(action=action_int)
        
        # Get action probabilities and value if possible
        try:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                # Check if the model has the required methods
                if hasattr(model.policy, 'get_distribution') and hasattr(model.policy, 'predict_values'):
                    distribution = model.policy.get_distribution(obs_tensor)
                    action_probs = distribution.distribution.probs.numpy().tolist()[0]
                    value = model.policy.predict_values(obs_tensor).numpy().tolist()[0][0]
                    
                    response.action_probs = action_probs
                    response.value = value
        except Exception as e:
            # If we can't get probabilities/value, just return action
            print(f"Warning: Could not get action probabilities or value: {e}")
            pass
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models."""
    models = []
    
    # Models in memory
    for name in running_models.keys():
        models.append({"name": name, "status": "loaded"})
    
    # Models on disk
    if os.path.exists(model_storage_path):
        for file in os.listdir(model_storage_path):
            if file.endswith(".zip"):
                name = file[:-4]  # Remove .zip extension
                if name not in running_models:
                    models.append({"name": name, "status": "saved"})
    
    return {"models": models}

@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a model."""
    # Remove from memory
    if model_name in running_models:
        del running_models[model_name]
    
    # Remove from disk
    model_path = os.path.join(model_storage_path, f"{model_name}.zip")
    if os.path.exists(model_path):
        os.remove(model_path)
        return {"message": f"Model {model_name} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "active_jobs": len([j for j in training_jobs.values() if j.status == "running"]),
        "loaded_models": len(running_models)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
