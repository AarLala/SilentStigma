FROM python:3.11-bullseye

WORKDIR /app

# Runtime libs for NumPy / SciPy / hdbscan wheels
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD ["python", "src/dashboard/app.py"]
