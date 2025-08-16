# Dockerfile
# Use an official lightweight Python base image
FROM python:3.11-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for compiling python libraries if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the source code and configuration files
COPY src/ /app/src/
COPY docs/ /app/docs/
COPY logs/ /app/logs/
COPY README.md /app/README.md

# Expose port 8000 for the FastAPI web server
EXPOSE 8000

# Command to run the Uvicorn FastAPI application
CMD ["uvicorn", "src.week6.api:app", "--host", "0.0.0.0", "--port", "8000"]
