FROM python:3.10-slim-bullseye

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        openjdk-17-jre-headless \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /app
COPY streamlit_dashboard/app.py .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        streamlit \
        redis \
        pyspark \
        pandas \
        pyarrow \
        boto3 \
        numpy \
        pillow \
        imageio[ffmpeg] \
        imageio-ffmpeg \
        torch

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
