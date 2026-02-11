FROM python:3.11-slim

WORKDIR /api

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8080"]