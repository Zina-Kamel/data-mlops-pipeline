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

st.set_page_config(page_title="Robot Dashboard", layout="wide")
st.title("Franka Arm Dashboard")

live_tab, stats_tab = st.tabs(["Live Data", "Statistics"])
r = redis.Redis(host="redis", port=6379)

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
                            aws_access_key_id="minio",
                            aws_secret_access_key="minio123"
                        )

                        filename = f"episode_{int(time.time())}.parquet"
                        s3.upload_fileobj(buffer, "robot-data", f"etl-output/dashboard/{filename}")

                        st.success(f"Saved episode to S3 as `{filename}`")
                        r.delete("franka_episode_buffer")

                    except Exception as e:
                        st.error(f"Failed to save episode: {str(e)}")

            with col_decline:
                if st.button("Discard Episode"):
                    r.delete("franka_episode_buffer")
                    st.info("Episode discarded.")
                    placeholder.empty() 

    else:
        st.warning("No episode data available yet.")

with stats_tab:
    st.write("Parquet-based stats")

    try:
        spark = SparkSession.builder \
        .appName("StatsViewer") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.2") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.access.key", "minio") \
        .config("spark.hadoop.fs.s3a.secret.key", "minio123") \
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


        spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", "minio")
        spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", "minio123")
        spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "http://minio:9000")
        spark._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
        spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

        df = spark.read.parquet("s3a://robot-data/etl-output/dashboard/")
        # conf = spark._jsc.hadoopConfiguration()
        # for entry in conf.iterator():
        #     st.text(f"{entry.getKey()} = {entry.getValue()}")

        print("Schema:")
        df.printSchema()
        st.write("Data loaded successfully from Parquet files.")
 
        st.write("Number of rows loaded:", df.count())
        st.write("Columns:", df.columns)
        st.bar_chart(df.select(col("`obs.gripper_width`")).toPandas())
        
    except Exception as e:
        st.error(f"Failed to load Parquet stats: {e}")
