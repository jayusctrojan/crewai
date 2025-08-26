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

EXPOSE 8000

# Debug version to see what's happening
CMD ["sh", "-c", "echo 'Container starting...' && echo 'Testing import...' && python -c 'import main; print(\"Import successful\")' && echo 'Starting uvicorn...' && uvicorn main:app --host 0.0.0.0 --port 8000"]