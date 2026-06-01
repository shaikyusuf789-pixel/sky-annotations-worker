FROM python:3.11-slim

# System dependencies:
#   ffmpeg        — video rendering
#   tesseract-ocr — slide OCR (pytesseract wrapper)
#   libcairo2     — CairoSVG SVG rasterization
#   fonts-dejavu  — ensures text renders correctly in cairosvg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    libsm6 \
    libxext6 \
    libcairo2 \
    libcairo2-dev \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    fonts-dejavu \
    gcc \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create tmp directory for rendering
RUN mkdir -p /tmp/render

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
