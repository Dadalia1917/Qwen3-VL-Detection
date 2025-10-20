# Qwen3-VL 通用目标检测系统

<p align="center">
    <img src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vllogo.png" width="400"/>
</p>

<p align="center">
    基于 Qwen3-VL 的通用目标检测系统，支持 2D/3D 目标检测
</p>

<p align="center">
    🤗 <a href="https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe">Hugging Face</a> | 
    🤖 <a href="https://modelscope.cn/collections/Qwen3-VL-5c7a94c8cb144b">ModelScope</a> | 
    📑 <a href="https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef">官方博客</a> | 
    📚 <a href="https://github.com/QwenLM/Qwen3-VL">官方仓库</a>
</p>

---

## 📖 项目简介

本项目基于阿里云通义千问视觉语言模型 **Qwen3-VL** 开发，提供了一套完整的通用目标检测解决方案。系统支持 **2D** 和 **3D** 目标检测，可应用于遥感图像、工业检测、仓储管理等多种场景。

### ✨ 主要特性

- 🎯 **2D 目标检测**：高精度 2D 边界框检测，支持置信度评分
- 🔮 **3D 目标检测**：3D 空间定位能力（持续优化中）
- 📊 **多种输出格式**：JSON、YOLO 格式、可视化图像
- ⚡ **批量处理**：支持大规模图像数据集的高效批处理
- 🛠️ **灵活配置**：可自定义模型路径、数据集目录和输出位置
- 💾 **内存优化**：自动图像缩放、动态显存管理、周期性内存清理

---

## 🚀 快速开始

### 环境要求

```bash
# Python 3.11+
pip install torch torchvision
pip install transformers>=4.57.0
pip install pillow opencv-python numpy matplotlib tqdm
```

### 模型下载

从 Hugging Face 或 ModelScope 下载 Qwen3-VL 模型到 `./models` 目录：

```bash
# Hugging Face
git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct ./models

# 或使用 ModelScope（国内推荐）
git clone https://modelscope.cn/Qwen/Qwen3-VL-4B-Instruct.git ./models
```

---

## 💡 使用方法

### 2D 目标检测

2D 检测系统提供精确的边界框定位和置信度评分，**适合各类图像场景**：

```bash
# 基础用法
python main2D.py

# 自定义路径
python main2D.py --model ./models --datasets ./your_data --outputs ./results/2D

# 指定设备
python main2D.py --device cuda
```

**输出目录结构：**
```
outputs/2D/
├── json/                    # JSON 格式检测结果
├── images/                  # 可视化检测结果
├── labels/                  # YOLO 格式标签
├── raw/                     # 模型原始输出
└── detection_summary.json   # 检测统计摘要
```

**检测结果示例：**
```json
{
  "image_name": "example.jpg",
  "detection_type": "2D",
  "original_size": {"width": 1920, "height": 1080},
  "detections": [
    {
      "bbox_absolute": [227, 203, 354, 367],
      "label": "object",
      "confidence": 0.95
    }
  ],
  "num_detections": 8
}
```

### 3D 目标检测（实验性功能）

3D 检测系统提供空间理解能力，适用于 **近距离室内/室外场景**。

> ⚠️ **注意**：3D 检测目前处于持续优化阶段，对于遥感/航拍图像效果有限，推荐使用真实相机内参以获得最佳效果。

```bash
# 基础用法（默认 FOV=60°）
python main3D.py

# 自定义视场角
python main3D.py --fov 70.0

# 完整配置
python main3D.py --model ./models --datasets ./your_data --outputs ./results/3D --fov 60
```

**输出目录结构：**
```
outputs/3D/
├── json/                    # 3D 边界框检测结果
├── images/                  # 3D 边界框可视化
├── visualizations/          # 增强可视化图
├── camera_params/           # 生成的相机参数
├── raw/                     # 模型原始输出
└── detection_summary.json   # 检测统计摘要
```

**3D 检测格式：**
```json
{
  "bbox_3d": [x, y, z, x_size, y_size, z_size, roll, pitch, yaw],
  "label": "object",
  "confidence": 0.92
}
```

参数说明：
- `x, y, z`：物体中心在相机坐标系中的位置（米）
- `x_size, y_size, z_size`：物体尺寸（米）
- `roll, pitch, yaw`：旋转角度（弧度）

---

## 📋 命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | `-m` | `./models` | Qwen3-VL 模型路径 |
| `--datasets` | `-d` | `./datasets` | 输入图像目录 |
| `--outputs` | `-o` | `./outputs/2D` 或 `./outputs/3D` | 输出目录 |
| `--device` | - | `auto` | 设备选择：`auto`、`cuda` 或 `cpu` |
| `--fov` | - | `60.0` | 相机视场角（度）- 仅 3D 检测 |

---

## 📊 性能说明

### 2D 检测
- ✅ **稳定可靠**：生产环境可用
- ⚡ **处理速度**：~30-60秒/张（取决于硬件）
- 🎯 **高精度**：适用于各类目标检测场景
- 📸 **广泛适用**：遥感图像、航拍影像、通用场景

### 3D 检测
- 🚧 **持续优化**：实验性功能，不断改进中
- ⏱️ **处理速度**：~5-10分钟/张
- 📸 **最佳效果**：需要真实相机内参
- 🏗️ **推荐场景**：室内/室外近距离场景

---

## 🎨 应用场景

- 🏭 **工业检测**：生产线目标检测与定位
- 🌍 **遥感分析**：卫星/航拍图像目标识别
- 📦 **仓储管理**：货物清点与追踪
- 🚗 **自动驾驶**：空间理解与障碍物检测
- 🏗️ **施工监控**：设备与物料追踪
- 🛡️ **安防监控**：目标识别与追踪

---

## 🔧 内存优化特性

系统内置多项内存优化功能，确保在有限资源下稳定运行：

- 📏 **自动缩放**：图像自动缩放至 1024px 以内
- 🎫 **令牌限制**：动态限制生成令牌数（1024）
- 🧹 **定期清理**：每处理 10 张图像自动清理 GPU 显存
- 💻 **精度支持**：支持 FP16/BF16 混合精度推理
- ⚡ **低显存模式**：`low_cpu_mem_usage=True` 优化加载

---

## 💡 使用技巧

### 2D 检测最佳实践
- ✅ 适用于任何类型的图像
- ✅ 自动处理各种分辨率
- ✅ 可在代码中调整置信度阈值
- ✅ 支持 NMS（非极大值抑制）去重

### 3D 检测最佳实践
- 📸 **提供真实相机参数**（如可用）以获得最佳效果
- 🎥 **使用近距离图像**，确保有清晰的深度信息
- 🔧 **根据相机调整 FOV**（默认 60°）
- 🏠 **适合室内/室外近场**场景
- ⚠️ 遥感/航拍图像由于缺乏深度信息，3D 效果有限

---

## ⚠️ 已知限制

- 3D 检测精度高度依赖相机参数和场景类型
- 遥感/航拍图像由于缺少深度信息，3D 检测结果可能不理想
- 3D 检测管道仍在持续优化中
- 建议对遥感应用使用 2D 检测

---

## 📂 项目结构

```
.
├── main2D.py              # 2D 目标检测主程序
├── main3D.py              # 3D 目标检测主程序
├── models/                # 模型文件目录
├── datasets/              # 输入图像目录
├── outputs/               # 输出结果目录
│   ├── 2D/               # 2D 检测结果
│   └── 3D/               # 3D 检测结果
├── detection_2d.log       # 2D 检测日志
├── detection_3d.log       # 3D 检测日志
└── README.md             # 项目说明文档
```

---

## 🙏 致谢

本项目基于阿里云通义千问团队开发的 [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) 模型构建。感谢 Qwen 团队的杰出工作！

### 相关资源

- 📚 [Qwen3-VL 官方文档](https://github.com/QwenLM/Qwen3-VL)
- 🤗 [Hugging Face 模型库](https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe)
- 🤖 [ModelScope 模型库](https://modelscope.cn/collections/Qwen3-VL-5c7a94c8cb144b)
- 📑 [技术博客](https://qwen.ai/blog)
- 📖 [Cookbook 示例](https://github.com/QwenLM/Qwen3-VL/tree/main/cookbooks)

---

## 📄 许可证

本项目遵循 [LICENSE](LICENSE) 文件中的许可协议。

Qwen3-VL 模型的使用需遵守阿里云通义千问的相关协议。

---

## 📮 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

## 📚 引用

如果本项目对您的研究有帮助，欢迎引用 Qwen3-VL 相关论文：

```bibtex
@article{Qwen2.5-VL,
  title={Qwen2.5-VL Technical Report},
  author={Bai, Shuai and Chen, Keqin and Liu, Xuejing and Wang, Jialin and others},
  journal={arXiv preprint arXiv:2502.13923},
  year={2025}
}

@article{Qwen2-VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and others},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```

---

<p align="center">
    ⭐ 如果觉得有用，请给个 Star！⭐
</p>
