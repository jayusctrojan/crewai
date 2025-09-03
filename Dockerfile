FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories for Studio UI (safe addition)
RUN mkdir -p static templates studio

EXPOSE 8000

# Back to the original working command
CMD ["python", "main.py"]