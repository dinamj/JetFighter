# Frontend
FROM node:20-slim AS frontend-build
WORKDIR /build
COPY jetfighter-monorepo/frontend/package.json jetfighter-monorepo/frontend/package-lock.json* ./
RUN npm install
COPY jetfighter-monorepo/frontend/ ./
RUN npm run build

# Backend + CLI
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CUDA 
# COPY jetfighter-monorepo/backend/requirements.txt ./requirements.txt
# RUN python -m pip install --no-cache-dir --upgrade pip \
#     && grep -v -E '^torch([<>=!~].*)?$' requirements.txt > requirements-no-torch.txt \
#     && pip install --no-cache-dir -r requirements-no-torch.txt pdf2image \
#     && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
#     torch torchvision torchaudio

# CPU
COPY jetfighter-monorepo/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt pdf2image


# Copy project
COPY models/ ./models/
COPY inference/ ./inference/
COPY jetfighter-monorepo/backend/ ./jetfighter-monorepo/backend/
COPY --from=frontend-build /build/dist/. ./jetfighter-monorepo/backend/static_frontend/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

WORKDIR /app/jetfighter-monorepo/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
