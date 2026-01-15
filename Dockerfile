# Use an official lightweight Python image
FROM python:3.11-slim

# Install system dependencies for OpenCV and Pillow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port (Cloud Run sets PORT env var)
EXPOSE 8080

# Run the application using gunicorn
# --timeout 0 disables the timeout of the workers to allow long-running video processing
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
