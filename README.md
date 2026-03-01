# AuraFace Backend | AuraFace 后端

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.109+-009?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/MediaPipe-0.10+-009?logo=google" alt="MediaPipe">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <strong>AI Face Analysis Backend</strong> — 478-point facial landmark detection, rule-based beauty scoring, and AI-powered analysis
</p>

<p align="center">
  <strong>AI 面部分析后端</strong> — 478点面部关键点检测、规则化颜值评分、AI驱动分析
</p>

---

## English | [中文](#中文)

### Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Core Algorithms](#core-algorithms)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Privacy](#privacy)
- [License](#license)

---

### Features

| Feature | Description |
|---------|-------------|
| **478-Point Detection** | Extract 478 facial landmarks using MediaPipe Face Mesh |
| **Face Shape Detection** | Automatically identify 6 face shapes: Round, Oval, Square, Heart, Diamond, Oblong |
| **Feature Scoring** | Calculate scores for eyes, nose, lips, eyebrows based on golden ratio |
| **Symmetry Analysis** | Calculate left-right face symmetry |
| **Measurements** | Detailed facial measurements in pixels |
| **Style Recommendations** | Hairstyle recommendations based on face shape |
| **AI Analysis** | Generate personalized comments using DeepSeek or OpenAI API |

---

### Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.12+ | Runtime |
| **FastAPI** | 0.109+ | Web Framework |
| **MediaPipe** | 0.10+ | Face Landmark Detection |
| **OpenCV** | - | Image Processing |
| **OpenAI API** | - | AI Generation |
| **DeepSeek API** | - | (Optional) AI Generation |

---

### Quick Start

```bash
# Clone the project
git clone https://github.com/lonnie08/auraface-api.git
cd auraface-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

After starting, visit:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

### API Endpoints

#### POST /api/analyze

Main face analysis endpoint.

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@your_photo.jpg"
```

Response:

```json
{
  "success": true,
  "face_shape": "Oval",
  "face_shape_confidence": 85.2,
  "shape_probabilities": {
    "oval": 0.45,
    "round": 0.20,
    "square": 0.15
  },
  "scores": {
    "eyebrows": 7.5,
    "eyes": 8.2,
    "lips": 7.8,
    "nose": 7.2
  },
  "characteristics": ["Balanced proportions", "Symmetrical features"],
  "style_recommendations": ["Almost any style works", "Try soft layers"],
  "annotated_image": "data:image/jpeg;base64,..."
}
```

#### POST /api/roast

AI-generated humorous analysis (supports both OpenAI and DeepSeek).

```bash
curl -X POST "http://localhost:8000/api/roast" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "gentle",
    "lang": "en",
    "provider": "openai",
    "face_data": {
      "face_shape": "Oval",
      "scores": {"eyebrows": 7.5, "eyes": 8.2, "lips": 7.8, "nose": 7.2},
      "proportions": {"face_ratio": 1.166},
      "characteristics": ["Balanced proportions"]
    }
  }'
```

#### POST /api/translate

Text translation endpoint.

```bash
curl -X POST "http://localhost:8000/api/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，世界！",
    "target_lang": "en",
    "source_lang": "auto"
  }'
```

Supported languages: en, zh, es, fr, de, it, pt, ru, ja, ko, ar, hi

#### GET /health

Health check endpoint.

---

### Core Algorithms

#### Face Shape Detection

Uses Gaussian distribution to calculate matching scores between facial proportions and each face shape:

```
l2w = face_length / cheek_width
fr = forehead_width / cheek_width
jr = jaw_width / cheek_width
```

#### Feature Scoring

| Feature | Basis | Ideal Value |
|---------|-------|-------------|
| Eyes | Eye distance / Face width | 0.48 |
| Lips | Lip width / Lip height | 2.0 |
| Nose | Nose width / Face width | 0.35 |
| Eyebrows | Eyebrow angle | 15° |

---

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | No | OpenAI API Key | - |
| `OPENAI_BASE_URL` | No | OpenAI API URL | https://api.openai.com/v1 |
| `OPENAI_MODEL` | No | OpenAI Model | gpt-4o-mini |
| `DEEPSEEK_API_KEY` | No | DeepSeek API Key | - |
| `DEEPSEEK_BASE_URL` | No | DeepSeek API URL | https://api.deepseek.com |
| `DEEPSEEK_MODEL` | No | DeepSeek Model | deepseek-chat |
| `ROAST_TIMEOUT_MS` | No | API Timeout (ms) | 12000 |

---

### Deployment

#### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t auraface-backend .
docker run -p 8000:8000 auraface-backend
```

#### Railway

1. Connect GitHub repository
2. Set environment variables
3. Deploy with: `python main.py`

#### VPS with Systemd

```bash
sudo nano /etc/systemd/system/auraface.service
```

```ini
[Unit]
Description=AuraFace Backend
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/auraface
ExecStart=/var/www/auraface/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

---

### Privacy

- **No Photo Storage** - All photos are processed in memory and released immediately after processing
- **Zero Data Retention** - No user photos or analysis results are stored
- **Secure Transmission** - HTTPS recommended for production deployment

---

### License

MIT License. See [LICENSE](LICENSE) for details.

---

### Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

### Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Face landmark detection
- [FastAPI](https://fastapi.tiangolo.com/) - High-performance web framework
- [OpenAI](https://openai.com/) - Large language model support
- [DeepSeek](https://www.deepseek.com/) - (Optional) AI model support

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/lonnie08">lonnie08</a>
</p>

---

# 中文

## 目录

- [功能特性](#功能特性-1)
- [技术栈](#技术栈-1)
- [快速开始](#快速开始-1)
- [API 端点](#api-端点-1)
- [核心算法](#核心算法-1)
- [环境变量](#环境变量-1)
- [部署指南](#部署指南-1)
- [隐私说明](#隐私说明-1)
- [许可证](#许可证-1)

---

## 功能特性

| 功能 | 描述 |
|------|------|
| **478点检测** | 使用 MediaPipe Face Mesh 提取478个面部关键点 |
| **脸型识别** | 自动识别6种常见脸型：圆形、椭圆形、方形、心形、菱形、长形 |
| **五官评分** | 基于黄金比例计算眼睛、鼻子、嘴唇、眉毛的评分 |
| **对称度分析** | 计算左右脸对称程度 |
| **测量数据** | 返回详细的面部测量数据（像素单位） |
| **发型推荐** | 根据脸型推荐适合的发型 |
| **AI分析** | 使用 DeepSeek 或 OpenAI API 生成个性化评语 |

---

## 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.12+ | 运行环境 |
| **FastAPI** | 0.109+ | Web 框架 |
| **MediaPipe** | 0.10+ | 面部关键点检测 |
| **OpenCV** | - | 图像处理 |
| **OpenAI API** | - | AI 生成 |
| **DeepSeek API** | - | （可选）AI 生成 |

---

## 快速开始

```bash
# 克隆项目
git clone https://github.com/lonnie08/auraface-api.git
cd auraface-api

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 启动服务
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

服务启动后访问：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

---

## API 端点

### POST /api/analyze

面部分析主端点。

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@your_photo.jpg"
```

### POST /api/roast

AI 吐槽生成端点（支持 OpenAI 和 DeepSeek）。

```bash
curl -X POST "http://localhost:8000/api/roast" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "gentle",
    "lang": "zh",
    "provider": "openai",
    "face_data": {
      "face_shape": "椭圆形",
      "scores": {"眉毛": 7.5, "眼睛": 8.2, "嘴唇": 7.8, "鼻子": 7.2},
      "proportions": {"脸型比例": 1.166},
      "characteristics": ["比例均衡"]
    }
  }'
```

### POST /api/translate

翻译端点。

```bash
curl -X POST "http://localhost:8000/api/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，世界！",
    "target_lang": "en",
    "source_lang": "auto"
  }'
```

支持语言：en, zh, es, fr, de, it, pt, ru, ja, ko, ar, hi

### GET /health

健康检查端点。

---

## 核心算法

### 脸型判断

使用高斯分布计算面部比例与各脸型的匹配度。

### 五官评分

| 五官 | 评分依据 | 理想值 |
|------|----------|--------|
| 眼睛 | 眼距/脸宽 | 0.48 |
| 嘴唇 | 唇宽/唇高 | 2.0 |
| 鼻子 | 鼻宽/脸宽 | 0.35 |
| 眉毛 | 眉毛角度 | 15° |

---

## 环境变量

| 变量 | 必需 | 描述 | 默认值 |
|------|------|------|--------|
| `OPENAI_API_KEY` | 否 | OpenAI API Key | - |
| `OPENAI_BASE_URL` | 否 | OpenAI API 地址 | https://api.openai.com/v1 |
| `OPENAI_MODEL` | 否 | OpenAI 模型 | gpt-4o-mini |
| `DEEPSEEK_API_KEY` | 否 | DeepSeek API Key | - |
| `DEEPSEEK_BASE_URL` | 否 | DeepSeek API 地址 | https://api.deepseek.com |
| `DEEPSEEK_MODEL` | 否 | DeepSeek 模型 | deepseek-chat |
| `ROAST_TIMEOUT_MS` | 否 | API 超时（毫秒）| 12000 |

---

## 部署指南

### Docker 部署

```bash
docker build -t auraface-backend .
docker run -p 8000:8000 auraface-backend
```

### Railway 部署

1. 连接 GitHub 仓库
2. 设置环境变量
3. 部署命令：`python main.py`

---

## 隐私说明

- **照片不存储** - 所有照片仅在内存中处理，处理完成后立即释放
- **零数据保留** - 不保存任何用户上传的照片或分析结果
- **传输加密** - 生产环境建议使用 HTTPS

---

## 许可证

MIT 许可证。详见 [LICENSE](LICENSE)。

---

## 贡献指南

欢迎提交 Pull Request！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

---

## 致谢

- [MediaPipe](https://mediapipe.dev/) - 面部关键点检测
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 Web 框架
- [OpenAI](https://openai.com/) - 大语言模型支持
- [DeepSeek](https://www.deepseek.com/) - （可选）AI 模型支持

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/lonnie08">lonnie08</a>
</p>
