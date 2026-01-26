FROM python:3.9-slim-bullseye

# 1. 换源并安装 Tesseract 和 OpenCV 的系统依赖 (libgl1, libglib2.0)
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5088
CMD ["gunicorn", "-w", "1", "--threads", "50", "-b", "0.0.0.0:5088", "--timeout", "120", "--access-logfile", "-", "app:app"]
