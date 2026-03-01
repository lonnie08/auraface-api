# AuraFace Backend

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.109+-009?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/MediaPipe-0.10+-009?logo=google" alt="MediaPipe">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <strong>AI Face Analysis Backend</strong> — 478-point facial landmark detection, rule-based beauty scoring, and AI-powered analysis
</p>

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [技术架构](#技术架构)
- [快速开始](#快速开始)
- [API 端点](#api-端点)
- [核心算法](#核心算法)
- [环境变量](#环境变量)
- [部署指南](#部署指南)
- [隐私说明](#隐私说明)
- [许可证](#许可证)

---

## 项目简介

AuraFace Backend 是一个基于 FastAPI 的面部分析后端服务，提供：

- 🎯 **478 点面部关键点检测** - 使用 MediaPipe Face Mesh
- 📊 **规则化颜值评分** - 基于面部比例和黄金分割
- 🤖 **AI 驱动的趣味分析** - 接入 DeepSeek 大模型生成个性化评语
- 🔒 **隐私优先** - 照片仅在内存中处理，不持久化存储

---

## 功能特性

| 功能 | 描述 |
|------|------|
| **面部关键点检测** | 使用 MediaPipe 提取 478 个面部关键点 |
| **脸型判断** | 自动识别 6 种常见脸型：圆形、椭圆形、方形、心形、菱形、长形 |
| **五官评分** | 基于黄金比例计算眼睛、鼻子、嘴唇、眉毛的评分 |
| **对称度分析** | 计算左右脸对称程度 |
| **测量数据** | 返回详细的面部测量数据（像素单位） |
| **发型推荐** | 根据脸型推荐适合的发型 |
| **AI 吐槽生成** | 接入 DeepSeek API 生成趣味评语（支持中英文） |

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        AuraFace Backend                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│   │   FastAPI   │───▶│  MediaPipe  │───▶│   Rules     │  │
│   │   (Web)     │    │  Face Mesh  │    │  Calculator │  │
│   └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                                       │          │
│         │                                       ▼          │
│         │                              ┌─────────────┐   │
│         │                              │  DeepSeek   │   │
│         │                              │  (Optional) │   │
│         │                              └─────────────┘   │
│         │                                       │          │
│         └──────────────▶ JSON Response ◀────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.12+ | 运行环境 |
| **FastAPI** | 0.109+ | Web 框架 |
| **MediaPipe** | 0.10+ | 面部关键点检测 |
| **OpenCV** | - | 图像处理 |
| **PyTorch** | - | (可选) AI 模型推理 |
| **DeepSeek API** | - | 大语言模型调用 |

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/lonnie08/auraface-api.git
cd auraface-api
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入必要的 API Key
```

### 5. 下载 MediaPipe 模型

首次运行会自动下载 MediaPipe 模型文件。

### 6. 启动服务

```bash
# 开发模式
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 或直接运行
python main.py
```

服务启动后访问：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

---

## API 端点

### POST /api/analyze

面部分析主端点。

**请求：**

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@your_photo.jpg"
```

**响应：**

```json
{
  "success": true,
  "face_shape": "Oval",
  "face_shape_confidence": 85.2,
  "shape_probabilities": {
    "oval": 0.45,
    "round": 0.20,
    "square": 0.15,
    "heart": 0.10,
    "diamond": 0.05,
    "oblong": 0.05
  },
  "scores": {
    "eyebrows": 7.5,
    "eyes": 8.2,
    "lips": 7.8,
    "nose": 7.2
  },
  "characteristics": [
    "Balanced proportions",
    "Symmetrical features",
    "High cheekbones",
    "Balanced forehead"
  ],
  "measurements": {
    "face_width_px": 480.5,
    "face_height_px": 560.2,
    "forehead_width_px": 280.3,
    "jaw_width_px": 320.8
  },
  "proportions": {
    "face_ratio": 1.166,
    "forehead_to_cheek_ratio": 0.58,
    "jaw_to_cheek_ratio": 0.67
  },
  "style_recommendations": [
    "Almost any style works",
    "Try soft layers or bangs",
    "Rectangular glasses"
  ],
  "annotated_image": "data:image/jpeg;base64,..."
}
```

### POST /api/roast

AI 吐槽生成端点。

**请求：**

```bash
curl -X POST "http://localhost:8000/api/roast" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "gentle",
    "lang": "en",
    "face_data": {
      "face_shape": "Oval",
      "scores": {
        "eyebrows": 7.5,
        "eyes": 8.2,
        "lips": 7.8,
        "nose": 7.2
      },
      "proportions": {
        "face_ratio": 1.166
      },
      "characteristics": ["Balanced proportions"]
    }
  }'
```

**响应：**

```json
{
  "comment": "Your oval face is like a perfectly balanced recipe - everything in moderation! Those symmetrical features are giving 'mathematically pleasing' vibes. Keep serving looks!"
}
```

### GET /health

健康检查端点。

**响应：**

```json
{
  "status": "ok",
  "beauty_model_loaded": false,
  "weights_path": "/path/to/weights/model.pth",
  "weights_exists": false
}
```

---

## 核心算法

### 1. 脸型判断

使用高斯分布计算面部比例与各脸型的匹配度：

```
l2w = face_length / cheek_width      # 长宽比
fr = forehead_width / cheek_width    # 额头比例
jr = jaw_width / cheek_width         # 下颌比例
```

每个脸型有理想比例值，计算实际值与理想值的偏差：

```python
def gauss(val, ideal, sigma):
    return math.exp(-((val - ideal) ** 2) / (2 * sigma ** 2))
```

### 2. 五官评分

| 五官 | 评分依据 | 理想值 |
|------|----------|--------|
| 眼睛 | 眼距 / 脸宽 | 0.48 |
| 嘴唇 | 唇宽 / 唇高 | 2.0 |
| 鼻子 | 鼻宽 / 脸宽 | 0.35 |
| 眉毛 | 眉毛角度 | 15° |

评分公式：

```python
score = max(min_score, min(max_score, 10 - abs(actual - ideal) * weight))
```

### 3. 对称度计算

比较左右对应关键点的坐标差异：

```python
symmetry_score = 100 - abs(left_eye_x - right_eye_x) * 100
```

---

## 环境变量

### AI 服务配置

| 变量 | 必需 | 描述 | 默认值 |
|------|------|------|--------|
| `DEEPSEEK_API_KEY` | 否 | DeepSeek API Key | - |
| `DEEPSEEK_BASE_URL` | 否 | DeepSeek API 地址 | https://api.deepseek.com |
| `DEEPSEEK_MODEL` | 否 | DeepSeek 模型名称 | deepseek-chat |
| `OPENAI_API_KEY` | 否 | OpenAI (ChatGPT) API Key | - |
| `OPENAI_BASE_URL` | 否 | OpenAI API 地址 | https://api.openai.com/v1 |
| `OPENAI_MODEL` | 否 | OpenAI 模型名称 | gpt-4o-mini |
| `ROAST_TIMEOUT_MS` | 否 | API 超时时间（毫秒）| 12000 |

### 支持的 AI 提供商

- **OpenAI (ChatGPT)** - 默认推荐，支持翻译功能
- **DeepSeek** - 备用选项

---

## API 端点 (更新)

### POST /api/translate

翻译端点，支持中英文等多种语言。

**请求：**

```bash
curl -X POST "http://localhost:8000/api/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，世界！",
    "target_lang": "en",
    "source_lang": "auto"
  }'
```

**响应：**

```json
{
  "translated_text": "Hello, World!",
  "detected_lang": "zh"
}
```

**支持的语言：**

| 代码 | 语言 |
|------|------|
| en | English |
| zh | Chinese |
| es | Spanish |
| fr | French |
| de | German |
| it | Italian |
| pt | Portuguese |
| ru | Russian |
| ja | Japanese |
| ko | Korean |
| ar | Arabic |
| hi | Hindi |

---

## 部署指南

### Docker 部署

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

### Railway 部署

1. 连接 GitHub 仓库
2. 设置环境变量
3. 部署命令：`python main.py`

### VPS 部署

```bash
# 使用 systemd 管理服务
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

## 隐私说明

### 数据处理

- 📷 **照片不存储** - 所有照片仅在内存中处理，处理完成后立即释放
- 🔒 **传输加密** - 建议使用 HTTPS 部署
- 🗑️ **零数据保留** - 不保存任何用户上传的照片或分析结果

### 安全措施

- CORS 配置允许跨域访问（生产环境建议限制）
- 输入验证防止恶意请求
- API 调用可选超时保护

---

## 许可证

本项目基于 **MIT 许可证** 开源。

```
MIT License

Copyright (c) 2024 AuraFace

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

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
- [DeepSeek](https://www.deepseek.com/) - 大语言模型支持

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/lonnie08">lonnie08</a>
</p>
