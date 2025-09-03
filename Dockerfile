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

# Expose both ports - 8000 for CrewAI API and 8501 for Archon Streamlit
EXPOSE 8000 8501

# Use the new multi-service startup script
CMD ["python", "startup.py"]