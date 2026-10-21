# Container image for Storyboard AI Studio on Google Cloud Run
FROM python:3.11-slim

# Install system dependencies including ffmpeg for video assembly
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Cloud Run expects the application to listen on the port specified by $PORT (defaults to 8080)
ENV PORT=8080
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "run_ui.py"]
