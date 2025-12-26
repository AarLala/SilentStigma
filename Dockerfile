FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD ["python", "src/dashboard/app.py"]
