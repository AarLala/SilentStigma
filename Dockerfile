FROM python:3.11-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --only-binary=:all: --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir langdetect==1.0.9

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD ["python", "src/dashboard/app.py"]
