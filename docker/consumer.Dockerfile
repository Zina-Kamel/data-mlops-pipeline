FROM python:3.10-slim
WORKDIR /app
COPY src/consumer/kafka_to_redis.py .
RUN pip install kafka-python redis
CMD ["python", "kafka_to_redis.py"]
