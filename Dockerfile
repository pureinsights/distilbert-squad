# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

ENV WORKERS=2
ENV THREADS=1
ENV TIMEOUT=30000
ENV MODELS_PATH='./src/models'

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY ./src ./src

# Install production dependencies.
# Copy application dependency manifests to the container image.
# Copying this separately prevents re-running pip install on every code change.

COPY requirements.txt .

RUN pip install -r requirements.txt

# Don't download the model. It's better to get it once and mount it multiple times.
# RUN python model.py

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.

EXPOSE 8080

ENTRYPOINT gunicorn --bind 0.0.0.0:8080 --workers ${WORKERS} --threads ${THREADS} src.main.app:app --timeout ${TIMEOUT}
