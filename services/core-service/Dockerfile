FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy shared package first (required dependency)
COPY ./shared /shared
RUN pip install --no-cache-dir -e /shared

# Copy application code (needed before installing package)
COPY ./core-service/ .

# Install service dependencies
RUN pip install --no-cache-dir -e .

# Make entrypoint script executable and move it to root
RUN chmod +x /app/entrypoint.sh && mv /app/entrypoint.sh /entrypoint.sh

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app /entrypoint.sh
USER app

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
# For local development: use uvicorn via entrypoint.sh
CMD ["/entrypoint.sh"]

# For Lambda deployment: AWS overrides CMD with Lambda runtime
# Lambda CMD will be: ["main.handler"] (configured in Lambda function settings)
