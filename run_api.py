#!/usr/bin/env python3
# run_api.py - Script to run the GRPO API server

import uvicorn
from api import app

if __name__ == "__main__":
    print("Starting GRPO API Server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "api:app",  # Use import string instead of app object
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload to avoid issues
            log_level="info"
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        # Fallback: run without reload
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
