#!/usr/bin/env python3
"""
Run script for the AI Video Editor backend server.
"""

import os
import argparse
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """
    Run the FastAPI server with the specified host and port.
    """
    parser = argparse.ArgumentParser(description="Run the AI Video Editor backend server")
    parser.add_argument(
        "--host", 
        type=str, 
        default=os.getenv("API_HOST", "0.0.0.0"),
        help="Host to run the server on"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.getenv("API_PORT", "8000")),
        help="Port to run the server on"
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    print(f"Starting AI Video Editor backend server on {args.host}:{args.port}")
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
