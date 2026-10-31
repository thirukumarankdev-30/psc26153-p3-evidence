FROM python:3.10.12-slim

# Set environment variables for Python in container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install pinned dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API application code only (data and model are mounted via volumes)
COPY api/ ./api/

# Expose default HTTP serving port
EXPOSE 8000

# Run FastAPI app with Uvicorn on 0.0.0.0:8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
