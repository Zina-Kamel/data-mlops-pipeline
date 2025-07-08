FROM python:3.10-slim
WORKDIR /app
COPY src/producer/mock_kafka_producer.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["python", "mock_kafka_producer.py"]