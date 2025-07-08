import json
import time
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import redis

print("Consumer Script Starting", flush=True)

consumer = None
while consumer is None:
    try:
        print("Trying to connect to Kafka broker...")
        consumer = KafkaConsumer(
            "robot-data",
            bootstrap_servers=["kafka:9092"],
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            group_id="robot-consumer-group",
            enable_auto_commit=True
        )
    except NoBrokersAvailable:
        print("Kafka broker not available yet. Retrying in 5 seconds...")
        time.sleep(5)

r = redis.Redis(host="redis", port=6379)

episode_buffer = []

for msg in consumer:
    data = msg.value
    print("Received message:", data)

    episode_buffer.append(data)

    if data.get("done"):
        print("Episode complete. Saving to Redis.")
        r.set("franka_episode_buffer", json.dumps(episode_buffer))
        episode_buffer = []  
