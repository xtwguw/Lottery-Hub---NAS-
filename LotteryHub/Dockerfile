FROM python:3.9-slim

WORKDIR /app

# 安装 OpenCV 和 OCR 系统依赖
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 5088

CMD ["gunicorn", "-w", "1", "--threads", "50", "-b", "0.0.0.0:5088", "--timeout", "60", "--access-logfile", "-", "app:app"]
