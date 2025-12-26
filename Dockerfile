FROM python:3.11-bullseye

WORKDIR /app

# Runtime scientific libs
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Force wheel-only install (no compiling)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --only-binary=:all: --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD ["python", "src/dashboard/app.py"]
