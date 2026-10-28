# Single-container build: React app is compiled and served by FastAPI.
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FRONTEND_DIST=/app/frontend/dist DB_PATH=/data/history.db HF_HOME=/opt/hf
WORKDIR /app/api
COPY api/requirements.txt api/requirements-leafprep.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Leaf preprocessing (BiRefNet-lite, CPU PyTorch); weights are baked into the image.
RUN pip install --no-cache-dir -r requirements-leafprep.txt --extra-index-url https://download.pytorch.org/whl/cpu  && python -c "from transformers import AutoModelForImageSegmentation as A; A.from_pretrained('ZhengPeng7/BiRefNet_lite', trust_remote_code=True)"  && chmod -R a+rX /opt/hf
COPY api/ ./
COPY --from=web /web/dist /app/frontend/dist
RUN useradd -m leaf && mkdir -p /data && chown leaf /data
USER leaf
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
# one worker: the leaf segmenter (~1 GB RAM) is loaded once per worker process
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1}"]
