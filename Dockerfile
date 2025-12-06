# Start from the official Airflow image
FROM apache/airflow:2.7.1-python3.10

# Switch to root to install system dependencies (needed for GeoPandas)
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user to install Python packages
USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt