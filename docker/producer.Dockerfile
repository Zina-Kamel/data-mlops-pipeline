FROM python:3.10-slim

WORKDIR /app

COPY src/producer/kafka_producer.py .

RUN pip install kafka-python \
    numpy \
    h5py \
CMD ["python", "kafka_producer.py"]