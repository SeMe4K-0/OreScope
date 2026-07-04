FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-сборка torch (лёгкий образ; для GPU см. README — базовый образ nvidia/cuda)
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# данные/выходы исключены через .dockerignore; models/ попадает в образ,
# если веса лежат рядом (иначе смонтировать томом: -v ./models:/app/models)
COPY . .

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/batches', timeout=4)"]
CMD ["python", "app.py"]
