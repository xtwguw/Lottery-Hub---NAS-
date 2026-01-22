# 🎰 Lottery Hub - NAS 彩票数据中心

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Lottery Hub 是一个专为 NAS（如飞牛、群晖、极空间）设计的 Docker 应用。它能自动同步中国福利彩票（双色球、七星彩）和体育彩票（大乐透）的历史数据，并提供强大的中奖查询、复式计算及可视化统计功能。

## ✨ 功能特性

- **数据全量同步**：自动抓取自彩票发行以来的所有历史开奖数据、奖池金额及中奖注数。
- **智能调度系统**：
  - 自动根据开奖日（如周二/四/日）晚间进入“追号模式”，每 5 分钟轮询直至更新。
  - 次日中午 12:00 自动补全详细奖金数据。
- **精准算奖**：
  - 支持 **复式投注** 自动拆分计算。
  - 支持大乐透 **追加模式** 计算。
  - 结果高亮显示：中奖号码红/蓝球高亮，未中奖号码置灰。
- **人性化界面**：
  - 双重倒计时：精确显示 **购票截止时间** 和 **开奖时间**。
  - 历史数据筛选：支持按年份、期数快速筛选。
  - 移动端适配：支持手机浏览器“添加到主屏幕”全屏运行。

## 🚀 快速部署 (Docker Compose)

### 1. 准备目录
在你的 NAS 上创建一个文件夹（例如 `/docker/lottery`），并确保其中包含一个空的 `data` 子文件夹。上传文件： 将上述 5 个文件（app.py, templates/index.html, docker-compose.yml, Dockerfile, requirements.txt）上传到 NAS 的 lottery 文件夹中。

### 2. 执行部署（SSH终端）：

Bash
cd /path/to/lottery  # 进入你的目录
docker-compose down  # 停止旧的
docker-compose up -d --build # 重新构建并启动
等待数据初始化： 容器启动后，init_db 会自动删除旧的数据库文件，然后开始重新下载所有历史数据。这个过程大约需要 10-30 秒。 你可以通过 docker logs -f lottery_helper 查看进度，直到出现 ✅ dlt 更新 xxx 条 字样。

访问使用：

PC/手机：访问 http://NAS_IP:5088。

飞牛 NAS：在 Docker 列表中点击容器旁边的“打开”按钮即可直达。
