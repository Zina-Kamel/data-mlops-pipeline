from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import *

obs_view_schema = StructType([
    StructField("color", ArrayType(ArrayType(ArrayType(DoubleType())))),  # 4x4x3
    StructField("depth", ArrayType(ArrayType(DoubleType())))              # 4x4
])

observation_schema = StructType([
    StructField("agentview", obs_view_schema),
    StructField("wristview", obs_view_schema),
    StructField("delta_time", DoubleType()),
    StructField("ee_pose", ArrayType(DoubleType())),    # 7D pose
    StructField("ee_quat", ArrayType(DoubleType())),    # 4D quat
    StructField("finger_left", DoubleType()),
    StructField("gripper_act", IntegerType()),
    StructField("gripper_width", DoubleType()),
    StructField("joint_pose", ArrayType(DoubleType())), # 7D
    StructField("joint_vel", ArrayType(DoubleType())),  # 7D
])

schema = StructType([
    StructField("obs", observation_schema),
    StructField("next_obs", observation_schema),
    StructField("actions", ArrayType(DoubleType())),     # 7D
    StructField("done", BooleanType()),
    StructField("language_prompts", StringType())
])

spark = SparkSession.builder \
    .appName("RobotETL") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2") \
    .getOrCreate()

hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.access.key", "minio")
hadoop_conf.set("fs.s3a.secret.key", "minio123")
hadoop_conf.set("fs.s3a.endpoint", "http://minio:9000")
hadoop_conf.set("fs.s3a.path.style.access", "true")
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "robot-data") \
    .load()

json_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

query = json_df.writeStream \
    .format("parquet") \
    .option("checkpointLocation", "/tmp/spark_checkpoint") \
    .option("path", "s3a://robot-data/etl-output/") \
    .outputMode("append") \
    .start()

query.awaitTermination()
