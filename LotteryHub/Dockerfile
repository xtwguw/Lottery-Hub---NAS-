# 使用轻量级 Python 镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口 (注意这里只是声明，实际由 gunicorn 绑定)
EXPOSE 5088

# 启动命令 (关键修改)
# -w 1: 1个 worker 进程 (节省内存)
# --threads 50: 每个 worker 50 个线程 (限制最大并发为 50)
# -b 0.0.0.0:5088: 绑定端口
# --timeout 60: 超时时间
# app:app : 指向 app.py 里的 app 对象
CMD ["gunicorn", "-w", "1", "--threads", "50", "-b", "0.0.0.0:5088", "--timeout", "60", "app:app"]