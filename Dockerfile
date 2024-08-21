FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install PyTorch with CPU-only support and other requirements
RUN pip install --no-cache-dir pytorch torchvision torchaudio cpuonly -c pytorch \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .
RUN rm -f .env

# Set the environment variable for Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

CMD ["python", "SRC/main.py"]