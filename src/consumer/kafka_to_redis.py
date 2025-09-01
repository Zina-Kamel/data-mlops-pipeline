import json
import time
import uuid
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
            group_id="test-group-244",
            enable_auto_commit=True,
            fetch_max_bytes=4 * 1024 * 1024,  # 4MB
            max_partition_fetch_bytes=2 * 1024 * 1024,
            consumer_timeout_ms=10000
        )
    except NoBrokersAvailable:
        print("Kafka broker not available yet. Retrying in 5 seconds...")
        time.sleep(5)

print("Connected to Kafka! Listening for messages...", flush=True)

r = redis.Redis(host="redis", port=6379, decode_responses=True)

episode_id = str(uuid.uuid4()) 
step = 0

episode_buffer = []  
try:
    print("Starting message loop...", flush=True)

    for msg in consumer:
        print("Received message")
        data = msg.value
        print(f"Received message with done = {data.get('done')} (type: {type(data.get('done'))})", flush=True)

        episode_buffer.append(data)

        if data.get("done"):
            print("Episode complete. Saving to Redis..")
            try:
                r.set("franka_episode_buffer", json.dumps(episode_buffer))
            except Exception as e:
                print("Redis save error:", e)
            episode_buffer.clear()
except Exception as e:
    print(f"Consumer loop crashed with error: {e}")
