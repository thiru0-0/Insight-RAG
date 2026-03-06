FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY tests/ tests/
COPY load_datasets.py .
COPY run_pipeline.py .
COPY .env.example .
COPY README.md .

# Copy dataset documents for full deployment
COPY docs/ docs/

# HF Spaces expects port 7860
ENV API_PORT=7860
EXPOSE 7860

# Start the server
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]
