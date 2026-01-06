#!/bin/bash
set -e

# Reinstall package in editable mode if pyproject.toml exists
if [ -f /app/pyproject.toml ]; then
    pip install --no-cache-dir -e /app
fi

# Reinstall shared package if it exists
if [ -d /shared ] && [ -f /shared/pyproject.toml ]; then
    pip install --no-cache-dir -e /shared
fi

# Start uvicorn
exec uvicorn main:app --host 0.0.0.0 --port 8001

