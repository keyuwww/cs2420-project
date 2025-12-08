#!/usr/bin/env python3

import google.generativeai as genai
import os
import sys

# Load environment variables from .env file
def load_env_file():
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value

# Load variables from .env file
load_env_file()

# Configure Gemini API
if "GEMINI_API_KEY" not in os.environ:
    print("Error: GEMINI_API_KEY not found in environment or .env file", file=sys.stderr)
    print("Please create a .env file in the project root with: GEMINI_API_KEY=your_api_key_here", file=sys.stderr)
    sys.exit(1)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# List available models
print("Available Gemini models:")
models = genai.list_models()
for m in models:
    print(f"- {m.name}")

