#!/bin/bash
set -e

# Start uvicorn (packages already installed in Dockerfile)
exec uvicorn main:app --host 0.0.0.0 --port 8001
