FROM python:3.11-bullseye

WORKDIR /app

# Runtime libs for NumPy / SciPy / hdbscan wheels
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install everything except langdetect using wheels only
RUN pip install --upgrade pip && \
    grep -v '^langdetect' requirements.txt > /tmp/req_no_langdetect.txt && \
    pip install --only-binary=:all: --no-cache-dir -r /tmp/req_no_langdetect.txt && \
    pip install --no-cache-dir langdetect==1.0.9

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD ["python", "src/dashboard/app.py"]
