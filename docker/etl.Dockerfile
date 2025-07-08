FROM bitnami/spark:3.1.2

COPY src/etl/etl_data_to_s3.py /app/etl_data_to_s3.py

CMD ["/opt/bitnami/spark/bin/spark-submit", "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2", "--conf", "spark.jars.ivy=/tmp/.ivy2", "/app/etl_data_to_s3.py"]


