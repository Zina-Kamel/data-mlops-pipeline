import json
import time
import h5py
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

print("Producer starting...")
HDF5_PATH = "/app/sample_data/frankaslideV4.hdf5"

def extract_obs(group, i):
    return {
        "agentview": {
            "color": group["agentview"]["color"][i].tolist(),
            "depth": group["agentview"]["depth"][i].tolist(),
        },
        "delta_time": float(group["delta_time"][i]),
        "ee_pose": group["ee_pose"][i].tolist(),
        "ee_quat": group["ee_quat"][i].tolist(),
        "gripper_state": int(group["gripper_state"][i]),
        "gripper_width": float(group["gripper_width"][i]),
        "joint_pose": group["joint_pose"][i].tolist(),
        "joint_vel": group["joint_vel"][i].tolist()
    }

producer = None
while producer is None:
    try:
        print("Trying to connect to Kafka broker...")
        producer = KafkaProducer(
            bootstrap_servers="kafka:9092",
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            batch_size=16384,    
            linger_ms=0,         
            max_request_size=20000000,
            request_timeout_ms=30000,  
            retries=5,
            retry_backoff_ms=2000
        )
        print("Connected to Kafka!")
    except NoBrokersAvailable:
        print("Kafka broker not available yet. Retrying in 5 seconds...")
        time.sleep(5)

def on_send_success(record_metadata):
    print(f"Message delivered to {record_metadata.topic} partition {record_metadata.partition} offset {record_metadata.offset}")

def on_send_error(excp):
    print('Message delivery failed:', excp)

print("Reading and sending messages one by one...")
start_time = time.time()
with h5py.File(HDF5_PATH, "r") as full_file:
    for i in range(len(full_file)):
        f = full_file[f"demo_{i}"]
        print(f"Processing demo_{i}...")
        actions = f["actions"]
        dones = f["done"]
        prompts = f["language_prompts"]
        obs = f["obs"]

        total = len(actions)
        for i in range(total):
            prompt = prompts[i]
            prompt = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
            
            done = bool(dones[i])

            msg = {
                "obs": extract_obs(obs, i),
                "actions": actions[i].tolist(),
                "done": done,
                "language_prompts": prompt
            }
            producer.send("robot-data", msg).add_callback(on_send_success).add_errback(on_send_error)

            if (i + 1) % 100 == 0:
                print(f"Sent {i+1}/{total} messages")
            
            if done:
                print(f"Episode done at step {i+1}., size {i}")
                break

producer.flush()
print(f"All messages sent in {time.time() - start_time:.2f} seconds.")
