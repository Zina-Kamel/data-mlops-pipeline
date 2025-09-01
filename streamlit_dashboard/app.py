import streamlit as st
import redis
import json
import time
import numpy as np
import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
import io
from PIL import Image
import base64
import io
import imageio
import random
import tempfile
import os
import subprocess
import torch
import torch.nn as nn
import boto3
import io
import json

from dotenv import load_dotenv

load_dotenv()

ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        return self.fc_out(x)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_best_model_from_s3(bucket="robot-data", artifact_prefix="ml", input_dim=22,
                            endpoint_url="http://minio:9000", aws_access_key=ACCESS_KEY,
                            aws_secret_key=SECRET_KEY):
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    best_model_key = f"{artifact_prefix}/best_model.json"
    buf = io.BytesIO()
    s3.download_fileobj(bucket, best_model_key, buf)
    buf.seek(0)
    best_info = json.load(buf)

    model_type = best_info["model_type"]
    model_key = f"{artifact_prefix}/{best_info['model_path']}"

    if model_type == "MLP":
        model = MLP(input_size=input_dim, output_size=1)
    elif model_type == "Transformer":
        model = TransformerModel(input_size=input_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    weights_buf = io.BytesIO()
    s3.download_fileobj(bucket, model_key, weights_buf)
    weights_buf.seek(0)
    model.load_state_dict(torch.load(weights_buf, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"Loaded best model from S3: {model_type} ({model_key})")
    return model

INPUT_DIM = 250  
best_model = load_best_model_from_s3(bucket="robot-data", artifact_prefix="ml", input_dim=INPUT_DIM)

def episode_to_tensor(episode):
    obs_cols = ["ee_pose", "ee_quat", "joint_pose", "joint_vel", "gripper_width"]

    features = []
    for step in episode:
        obs = step.get("obs", {})
        step_features = []
        for col in obs_cols:
            val = obs.get(col)
            if isinstance(val, (list, np.ndarray)):
                step_features.extend(np.array(val, dtype=np.float32).flatten())
            else:
                step_features.append(float(val) if val is not None else 0.0)
        features.append(step_features)

    input_array = np.array(features[-1], dtype=np.float32)

    if len(input_array) < INPUT_DIM:
        input_array = np.pad(input_array, (0, INPUT_DIM - len(input_array)), mode='constant')
    else:
        input_array = input_array[:INPUT_DIM]

    input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return input_tensor


def recommend_episode(episode):
    if best_model is None:
        return "Model not loaded"

    x = episode_to_tensor(episode)  
    with torch.no_grad():
        pred = best_model(x)
    score = pred.mean().item()
    return "Accept" if score > 0.5 else "Reject"



st.set_page_config(page_title="Robot Dashboard", layout="wide")
st.title("Franka Arm Dashboard")

live_tab, stats_tab, rlhf_tab = st.tabs(["Live Data", "Statistics", "RLHF Episodes"])
r = redis.Redis(host="redis", port=6379)

def display_video(video_bytes, width=320, height=240):
    if video_bytes is None:
        st.warning("No video to display")
        return

    video_html = f"""
    <video width="{width}" height="{height}" controls>
        <source src="data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}" type="video/mp4">
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)


with live_tab:
    placeholder = st.empty()
    data_json = r.get("franka_episode_buffer")

    if data_json:
        episode = json.loads(data_json)

        with placeholder.container():
            st.subheader(f"Episode with {len(episode)} steps")

            for i, step in enumerate(episode):
                with st.expander(f"Step {i+1}"):
                    obs = step.get("obs", {})
                    next_obs = step.get("next_obs", {})

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("EE Pose:", obs.get("ee_pose"))
                        st.write("EE Quaternion:", obs.get("ee_quat"))
                        st.write("Gripper Width:", obs.get("gripper_width"))
                        st.write("Finger Left:", obs.get("finger_left"))
                    with col2:
                        st.write("Joint Pose:", obs.get("joint_pose"))
                        st.write("Joint Velocity:", obs.get("joint_vel"))

                    st.write("Actions:", step.get("actions"))
                    st.success(f'Prompt: {step.get("language_prompts")}')
                    if step.get("done"):
                        st.warning("Episode DONE")
                    else:
                        st.info("Episode ACTIVE")

            st.markdown("---")
            st.subheader("Episode Observation Gallery")

            images = []
            for step in episode:
                img_data = step.get("obs", {}).get("agentview", {}).get("color")
                if img_data:
                    arr = np.array(img_data, dtype=np.uint8)
                    img = Image.fromarray(arr)
                    images.append(img)

            if images:
                with st.container():
                    gallery_html = """
                    <div style='
                        overflow-x: auto;
                        white-space: nowrap;
                        padding: 10px;
                        border: 1px solid #ddd;
                        max-height: 220px;
                        overflow-x: scroll;
                    '>
                    """
                    for img in images:
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        img_str = base64.b64encode(buffer.getvalue()).decode()
                        gallery_html += f"<img src='data:image/png;base64,{img_str}' width='200' style='display:inline-block; margin-right:10px;'/>"
                    gallery_html += "</div>"
                    st.markdown(gallery_html, unsafe_allow_html=True)

                st.subheader("Episode Video Preview")

                with tempfile.TemporaryDirectory() as tmpdir:
                    for i, img in enumerate(images):
                        frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
                        img.save(frame_path)

                    video_path = os.path.join(tmpdir, "episode.mp4")
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-framerate", "5",
                        "-i", os.path.join(tmpdir, "frame_%04d.png"),
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        video_path
                    ]
                    subprocess.run(cmd, check=True)

                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    display_video(video_bytes, width=420, height=340)

                        
            else:
                st.info("No images available to render gallery or video.")
            
            recommendation = recommend_episode(episode)
            st.info(f"Model Recommendation: **{recommendation}**")

            col_accept, col_decline = st.columns(2)
            with col_accept:
                if st.button("Keep Episode"):
                    try:
                        df = pd.json_normalize(episode)
                        table = pa.Table.from_pandas(df)
                        buffer = io.BytesIO()
                        pq.write_table(table, buffer)
                        buffer.seek(0)

                        s3 = boto3.client(
                            's3',
                            endpoint_url="http://minio:9000",
                            aws_access_key_id=ACCESS_KEY,
                            aws_secret_access_key=SECRET_KEY
                        )

                        filename = f"episode_{int(time.time())}.parquet"
                        buffer.seek(0)
                        data = buffer.getvalue()  

                        s3.upload_fileobj(io.BytesIO(data), "robot-data", f"etl-output/accepted_episodes/{filename}")

                        s3.upload_fileobj(io.BytesIO(data), "robot-data", f"etl-output/rlhf_episodes/{filename}")

                        st.success(f"Saved episode to S3 as `{filename}`")
                        r.delete("franka_episode_buffer")

                    except Exception as e:
                        st.error(f"Failed to save episode: {str(e)}")

            with col_decline:
                if st.button("Discard Episode"):
                    df = pd.json_normalize(episode)
                    table = pa.Table.from_pandas(df)
                    buffer = io.BytesIO()
                    pq.write_table(table, buffer)
                    buffer.seek(0)

                    s3 = boto3.client(
                        's3',
                        endpoint_url="http://minio:9000",
                        aws_access_key_id=ACCESS_KEY,
                        aws_secret_access_key=SECRET_KEY
                    )

                    filename = f"episode_{int(time.time())}.parquet"
                    s3.upload_fileobj(buffer, "robot-data", f"etl-output/rlhf_episodes/{filename}")
                    st.success(f"Saved episode to S3 as `{filename}`")  
                    r.delete("franka_episode_buffer")
                    st.info("Episode discarded.")
                    placeholder.empty()
    else:
        st.warning("No episode data available yet.")

SEEN_EPISODES_KEY = "rlhf_preferences/seen_episode_keys.json"

def load_seen_episodes():
    try:
        seen_buf = io.BytesIO()
        s3.download_fileobj("robot-data", SEEN_EPISODES_KEY, seen_buf)
        seen_buf.seek(0)
        return json.load(seen_buf)
    except:
        return []  

def save_seen_episodes(seen):
    buf = io.BytesIO()
    buf.write(json.dumps(seen).encode())
    buf.seek(0)
    s3.upload_fileobj(buf, "robot-data", SEEN_EPISODES_KEY)

def load_episode_from_s3(key, bucket="robot-data", endpoint_url="http://minio:9000"):
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    
    buffer = io.BytesIO()
    s3.download_fileobj(bucket, key, buffer)
    buffer.seek(0)
    
    table = pq.read_table(buffer)
    return table.to_pandas()

def create_episode_video(df, fps=5):
    images = []

    for i, row in df.iterrows():
        img_data = row.get("obs.agentview.color")
        if img_data is None:
            st.write(f"Step {i}: No image data")
            continue

        try:
            rows = [np.stack(row, axis=0) for row in img_data]
            img_array = np.stack(rows, axis=0).astype(np.uint8)
            images.append(img_array)
        except Exception as e:
            st.write(f"Step {i}: Failed to convert image: {e}")

    if not images:
        st.write("No images collected")
        return None

    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmpfile:
        writer = imageio.get_writer(tmpfile.name, fps=fps)
        for img in images:
            writer.append_data(img)
        writer.close()

        tmpfile.seek(0)
        video_bytes = tmpfile.read()

    return video_bytes



with rlhf_tab:
    st.title("RLHF Preference Selection")
    
    s3 = boto3.client(
        's3',
        endpoint_url="http://minio:9000",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    SEEN_EPISODES_KEY = "rlhf_preferences/seen_episode_keys.json"

    def load_seen_episodes(s3):
        try:
            s3.head_object(Bucket="robot-data", Key=SEEN_EPISODES_KEY)
            buf = io.BytesIO()
            s3.download_fileobj("robot-data", SEEN_EPISODES_KEY, buf)
            buf.seek(0)
            return json.load(buf)
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] in ("404", "NoSuchKey"):
                empty_list = []
                save_seen_episodes(s3, empty_list)
                return empty_list
            else:
                raise

    def save_seen_episodes(s3, seen):
        buf = io.BytesIO()
        buf.write(json.dumps(seen).encode())
        buf.seek(0)
        s3.upload_fileobj(buf, "robot-data", SEEN_EPISODES_KEY)

    seen_episode_keys = load_seen_episodes(s3)

    response = s3.list_objects_v2(Bucket="robot-data", Prefix="etl-output/rlhf_episodes/")
    all_episode_keys = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.parquet')]
    new_episode_keys = sorted(list(set(all_episode_keys) - set(seen_episode_keys)))

    if not new_episode_keys:
        st.info("No new episodes to label. All caught up!")
    else:
        if "episode_pairs" not in st.session_state:
            shuffled = new_episode_keys.copy()
            random.shuffle(shuffled)
            st.session_state.episode_pairs = list(zip(shuffled[::2], shuffled[1::2]))

        if "updated_seen" not in st.session_state:
            st.session_state.updated_seen = seen_episode_keys.copy()

        for ep_a_key, ep_b_key in st.session_state.episode_pairs[:3]:
            col1, col2 = st.columns(2)
            safe_key = f"{os.path.basename(ep_a_key)}_{os.path.basename(ep_b_key)}"

            with col1:
                st.subheader(f"A: `{os.path.basename(ep_a_key)}`")
                df_a = load_episode_from_s3(ep_a_key)
                video_a = create_episode_video(df_a)
                if video_a:
                    st.video(video_a, format="video/mp4")
                else:
                    st.warning("No video content to display")

            with col2:
                st.subheader(f"B: `{os.path.basename(ep_b_key)}`")
                df_b = load_episode_from_s3(ep_b_key)
                video_b = create_episode_video(df_b)
                if video_b:
                    st.video(video_b, format="video/mp4")
                else:
                    st.warning("No video content to display")

            radio_key = f"choice_{safe_key}"
            if radio_key not in st.session_state:
                st.session_state[radio_key] = ep_a_key  

            st.session_state[radio_key] = st.radio(
                "Which do you prefer?",
                options=[ep_a_key, ep_b_key],
                key=f"radio_{safe_key}",
                format_func=lambda k: "Episode A" if k == ep_a_key else "Episode B",
                index=0 if st.session_state[radio_key] == ep_a_key else 1
            )

            choice = st.session_state[radio_key]
            st.write("Current choice:", "Episode A" if choice == ep_a_key else "Episode B")

            if st.button("Submit Preference", key=f"submit_{safe_key}"):
                st.write("Saving preference...")
                try:
                    preference = {
                        "episode_a": ep_a_key,
                        "episode_b": ep_b_key,
                        "preferred": choice,
                        "timestamp": int(time.time())
                    }

                    json_buf = io.BytesIO()
                    json_buf.write(json.dumps(preference).encode())
                    json_buf.seek(0)
                    pref_filename = f"preference_{int(time.time())}.json"
                    s3.upload_fileobj(json_buf, "robot-data", f"rlhf_preferences/{pref_filename}")
                    st.success(f"Preference saved as `{pref_filename}`!")

                    st.session_state.updated_seen.extend([ep_a_key, ep_b_key])
                    buf = io.BytesIO()
                    buf.write(json.dumps(st.session_state.updated_seen).encode())
                    buf.seek(0)
                    s3.upload_fileobj(buf, "robot-data", SEEN_EPISODES_KEY)
                    st.write("Updated seen episodes successfully.")

                except Exception as e:
                    st.error(f"Failed to save preference: {e}")




with stats_tab:

    try:
        spark = SparkSession.builder \
        .appName("StatsViewer") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.2") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.access.key", ACCESS_KEY) \
        .config("spark.hadoop.fs.s3a.secret.key", SECRET_KEY) \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "30000") \
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "30000") \
        .config("spark.hadoop.fs.s3a.connection.timeout", "200000") \
        .config("spark.hadoop.fs.s3a.connection.ttl", "300000") \
        .config("spark.hadoop.fs.s3a.retry.throttle.interval", "100") \
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400000") \
        .config("spark.hadoop.fs.s3a.assumed.role.session.duration", "1800000") \
        .config("spark.hadoop.fs.s3a.readahead.range", "65536") \
        .config("spark.hadoop.fs.s3a.vectored.read.min.seek.size", "131072") \
        .config("spark.hadoop.fs.s3a.multipart.size", str(64 * 1024 * 1024)) \
        .config("spark.hadoop.fs.s3a.multipart.threshold", str(128 * 1024 * 1024)) \
        .config("spark.hadoop.yarn.resourcemanager.delegation-token-renewer.thread-retry-interval", "60000") \
        .config("spark.hadoop.yarn.resourcemanager.delegation-token-renewer.thread-timeout", "60000") \
        .config("spark.hadoop.yarn.router.subcluster.cleaner.interval.time", "60000") \
        .config("spark.hadoop.yarn.federation.state-store.heartbeat.initial-delay", "30000") \
        .config("spark.hadoop.yarn.federation.gpg.subcluster.heartbeat.expiration-ms", "1800000") \
        .config("spark.hadoop.yarn.federation.state-store.sql.idle-time-out", "600000") \
        .config("spark.hadoop.yarn.federation.gpg.policy.generator.interval", "3600000") \
        .config("spark.hadoop.yarn.federation.amrmproxy.register.uam.interval", "100") \
        .config("spark.hadoop.yarn.apps.cache.expire", "30000") \
        .config("spark.hadoop.yarn.nodemanager.log.delete.threshold", str(100 * 1024 * 1024 * 1024)) \
        .config("spark.hadoop.yarn.federation.state-store.sql.max-life-time", "1800000") \
        .config("spark.hadoop.yarn.federation.gpg.webapp.read-timeout", "30000") \
        .config("spark.hadoop.yarn.dispatcher.print-thread-pool.keep-alive-time", "10000") \
        .config("spark.hadoop.yarn.federation.state-store.sql.conn-time-out", "10000") \
        .config("spark.hadoop.yarn.federation.gpg.subcluster.cleaner.interval-ms", "-1") \
        .config("spark.hadoop.yarn.federation.gpg.application.cleaner.interval-ms", "-1000") \
        .config("spark.hadoop.yarn.federation.gpg.webapp.connect-timeout", "30000") \
        .config("spark.hadoop.yarn.router.submit.interval.time", "10") \
        .config("spark.hadoop.yarn.router.subcluster.heartbeat.expiration.time", "1800000") \
        .config("spark.hadoop.yarn.federation.state-store.clean-up-retry-sleep-time", "1000") \
        .config("spark.hadoop.hadoop.service.shutdown.timeout", "30000") \
        .config("spark.hadoop.hadoop.security.groups.shell.command.timeout", "0") \
        .config("spark.hadoop.yarn.resourcemanager.delegation.token.remove-scan-interval", "3600000") \
        .config("spark.hadoop.yarn.router.asc-interceptor-max-size", str(1 * 1024 * 1024)) \
        .config("spark.hadoop.fs.azure.sas.expiry.period", str(90 * 24 * 60 * 60 * 1000)) \
        .getOrCreate()


        spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", ACCESS_KEY)
        spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", SECRET_KEY)
        spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "http://minio:9000")
        spark._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
        spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

        df = spark.read.parquet("s3a://robot-data/etl-output/dashboard/dashboard.parquet") 

        print("Schema:")
        df.printSchema()
        st.write("Data loaded successfully from Parquet files.")
 
        st.write("Number of rows loaded:", df.count())
        st.write("Columns:", df.columns)
        st.bar_chart(df.select(col("`obs.gripper_width`")).toPandas())
        st.bar_chart(df.select(col("`language_prompts`")).toPandas())
        
    except Exception as e:
        st.error(f"Failed to load Parquet stats: {e}")
