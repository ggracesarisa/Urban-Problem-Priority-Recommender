FROM apache/airflow:2.7.1-python3.10

USER root

# 1. Install System Dependencies (GDAL for Geopandas)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# 2. Install LIGHTWEIGHT libraries first
RUN pip install --no-cache-dir \
    pandas \
    requests \
    shapely \
    scikit-learn \
    openai \
    streamlit \
    pyarrow \
    fastparquet \
    psycopg2-binary \
    rapidfuzz \
    haversine

# 3. Install HEAVY libraries separately (One by one to prevent Crash)
# Installing PyTorch CPU version explicitly to save 2GB RAM
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Geopandas (Requires GDAL we installed above)
RUN pip install --no-cache-dir geopandas

# Install NLP Libraries (The heaviest part)
RUN pip install --no-cache-dir transformers sentencepiece pythainlp

# 4. Final check
COPY requirements.txt /requirements.txt