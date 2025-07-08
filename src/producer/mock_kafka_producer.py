import json
import time
import h5py
import numpy as np
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

HDF5_PATH = "/app/sample_data/demo_0001.hdf5"

def load_hdf5_data(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        print("loading data")
        actions = f["actions"][:]
        dones = f["done"][:]
        prompts = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in f["language_prompts"][:]]
        obs = f["obs"]
        next_obs = f["next_obs"]

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

        messages = []
        for i in range(len(actions)):
            msg = {
                "obs": extract_obs(obs, i),
                "next_obs": extract_obs(next_obs, i),
                "actions": actions[i].tolist(),
                "done": bool(dones[i]),
                "language_prompts": prompts[i]
            }
            messages.append(msg)
        return messages

producer = None
while producer is None:
    try:
        print("Trying to connect to Kafka broker...")
        producer = KafkaProducer(
            bootstrap_servers="kafka:9092",
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
    except NoBrokersAvailable:
        print("Kafka broker not available yet. Retrying in 5 seconds...")
        time.sleep(5)

messages = load_hdf5_data(HDF5_PATH)

for i, msg in enumerate(messages):
    print(f"Sending step {i+1}/{len(messages)} to Kafka...")
    producer.send("robot-data", msg)
    print(msg)
    time.sleep(0.1) 
