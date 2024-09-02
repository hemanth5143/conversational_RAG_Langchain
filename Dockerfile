FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install PyTorch with CPU-only support and other requirements
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .
RUN rm -f .env

# Set the environment variable for Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "SRC/main.py", "--server.port=5000", "--server.address=0.0.0.0"]
