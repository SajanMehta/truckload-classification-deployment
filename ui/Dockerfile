FROM python:3.11-slim

WORKDIR /ui

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ui.py .

CMD ["sh", "-c", "python ui.py"]