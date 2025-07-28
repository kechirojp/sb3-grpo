# api_client_example.py - Example client for the GRPO API

import requests
import time
import json
from typing import Dict, Any

class GRPOAPIClient:
    """Client for interacting with the GRPO API server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def start_training(self, config: Dict[str, Any]) -> str:
        """Start a new training job."""
        response = self.session.post(f"{self.base_url}/train", json=config)
        response.raise_for_status()
        return response.json()["job_id"]
    
    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a training job."""
        response = self.session.get(f"{self.base_url}/training/{job_id}/status")
        response.raise_for_status()
        return response.json()
    
    def wait_for_training(self, job_id: str, check_interval: int = 5) -> Dict[str, Any]:
        """Wait for training to complete, checking status periodically."""
        while True:
            status = self.get_training_status(job_id)
            print(f"Training progress: {status['progress']:.2%} ({status['current_timesteps']}/{status['total_timesteps']} timesteps)")
            
            if status["status"] in ["completed", "failed", "stopped"]:
                return status
            
            time.sleep(check_interval)
    
    def stop_training(self, job_id: str) -> Dict[str, Any]:
        """Stop a running training job."""
        response = self.session.post(f"{self.base_url}/training/{job_id}/stop")
        response.raise_for_status()
        return response.json()
    
    def list_jobs(self) -> Dict[str, Any]:
        """List all training jobs."""
        response = self.session.get(f"{self.base_url}/training/jobs")
        response.raise_for_status()
        return response.json()
    
    def predict(self, model_name: str, observation: list, deterministic: bool = True) -> Dict[str, Any]:
        """Make a prediction using a trained model."""
        data = {
            "model_name": model_name,
            "observation": observation,
            "deterministic": deterministic
        }
        response = self.session.post(f"{self.base_url}/inference", json=data)
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """Delete a model."""
        response = self.session.delete(f"{self.base_url}/models/{model_name}")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API server health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the GRPO API client."""
    # Initialize client
    client = GRPOAPIClient()
    
    # Check server health
    try:
        health = client.health_check()
        print("✅ Server is healthy:", health)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running at http://localhost:8000")
        return
    
    # Example 1: Start training
    print("\n=== Starting Training ===")
    training_config = {
        "env_name": "CartPole-v1",
        "total_timesteps": 5000,
        "learning_rate": 3e-4,
        "n_steps": 32,
        "batch_size": 64,
        "n_epochs": 10,
        "reward_function_type": "optimized_cartpole",
        "model_name": "test_model"
    }
    
    job_id = client.start_training(training_config)
    print(f"🚀 Training started with job ID: {job_id}")
    
    # Monitor training progress
    print("\n=== Monitoring Training ===")
    final_status = client.wait_for_training(job_id)
    print(f"✅ Training finished with status: {final_status['status']}")
    
    if final_status["status"] == "completed":
        # Example 2: Use trained model for inference
        print("\n=== Testing Inference ===")
        
        # Test observation for CartPole (cart position, cart velocity, pole angle, pole velocity)
        test_observation = [0.1, 0.0, 0.05, 0.0]
        
        prediction = client.predict("test_model", test_observation)
        print(f"🎯 Prediction: action={prediction['action']}")
        if prediction.get("action_probs"):
            print(f"   Action probabilities: {prediction['action_probs']}")
        if prediction.get("value"):
            print(f"   State value: {prediction['value']:.3f}")
    
    # Example 3: List all models
    print("\n=== Available Models ===")
    models = client.list_models()
    for model in models["models"]:
        print(f"📦 {model['name']} ({model['status']})")
    
    # Example 4: List all jobs
    print("\n=== Training Jobs ===")
    jobs = client.list_jobs()
    for job in jobs["jobs"]:
        print(f"🔧 {job['job_id']}: {job['status']} ({job['progress']:.1%})")


if __name__ == "__main__":
    main()
