# MMPose 21点骨骼模型

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.8+-orange.svg)](https://pytorch.org/)

## 📖 简介

本项目基于 [MMPose](https://github.com/open-mmlab/mmpose) 框架，扩展了标准的 COCO 17点关键点模型，新增了4个脚部关键点（左右脚跟和左右脚尖），形成了**21点骨骼模型**。该模型专门针对双杠动作姿态估计进行了优化和训练。

### 主要特性

- ✅ 扩展了 COCO 17点模型，新增4个脚部关键点
- ✅ 支持完整的21点骨骼连接
- ✅ 基于 MMPose 框架，易于使用和扩展
- ✅ 提供预训练权重

## 🎯 关键点定义

本模型包含21个关键点，定义如下：

| ID | 名称 | 类型 | 说明 |
|----|------|------|------|
| 0-16 | COCO标准17点 | - | 与COCO数据集一致 |
| 17 | left_heel | lower | 左脚跟 |
| 18 | right_heel | lower | 右脚跟 |
| 19 | left_foot | lower | 左脚尖 |
| 20 | right_foot | lower | 右脚尖 |

### 关键点可视化

```
头部区域 (0-4):
  0: nose
  1: left_eye   2: right_eye
  3: left_ear   4: right_ear

上身区域 (5-12):
  5: left_shoulder   6: right_shoulder
  7: left_elbow      8: right_elbow
  9: left_wrist     10: right_wrist
 11: left_hip       12: right_hip

下身区域 (13-20):
 13: left_knee      14: right_knee
 15: left_ankle     16: right_ankle
 17: left_heel      18: right_heel
 19: left_foot      20: right_foot
```

## 🚀 快速开始

### 环境要求

- Python >= 3.7
- PyTorch >= 1.8
- CUDA >= 10.2 (如使用GPU)
- mmcv-full
- mmengine
- mmdet (可选，用于目标检测)

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/mmpose-21keypoints.git
cd mmpose-21keypoints
```

#### 2. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装 mmcv (根据你的CUDA版本选择)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch1.13/index.html

# 以开发模式安装 mmpose
pip install -e .
```

#### 3. 下载预训练权重

**方式一：从 GitHub Releases 下载（推荐）**

```bash
# 访问 Releases 页面下载权重文件
# https://github.com/yourusername/mmpose-21keypoints/releases

# 或使用 wget
wget https://github.com/yourusername/mmpose-21keypoints/releases/download/v1.0/best_coco_AP_epoch_110.pth
mkdir -p checkpoints
mv best_coco_AP_epoch_110.pth checkpoints/
```

**方式二：从云存储下载**

```bash
# 根据 README 中的云存储链接下载
# 例如：Google Drive, 百度网盘等
```

## 💻 使用方法

### 推理示例

#### 单张图片推理

```python
from mmpose.apis import MMPoseInferencer

# 创建推理器
inferencer = MMPoseInferencer(
    pose2d='configs/body_2d_keypoint/your_config.py',
    pose2d_weights='checkpoints/best_coco_AP_epoch_110.pth'
)

# 推理单张图片
result = inferencer('path/to/image.jpg', vis_out_dir='vis_results')
```

#### 视频推理

```python
from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer(
    pose2d='configs/body_2d_keypoint/your_config.py',
    pose2d_weights='checkpoints/best_coco_AP_epoch_110.pth'
)

# 推理视频
result = inferencer('path/to/video.mp4', vis_out_dir='vis_results')
```

#### 使用命令行工具

```bash
python demo/image_demo.py \
    path/to/image.jpg \
    configs/body_2d_keypoint/your_config.py \
    checkpoints/best_coco_AP_epoch_110.pth \
    --out-file vis_results/result.jpg
```

### 训练模型

```bash
# 单GPU训练
python tools/train.py configs/body_2d_keypoint/your_config.py

# 多GPU训练
bash tools/dist_train.sh configs/body_2d_keypoint/your_config.py 4
```

### 评估模型

```bash
python tools/test.py \
    configs/body_2d_keypoint/your_config.py \
    checkpoints/best_coco_AP_epoch_110.pth \
    --eval mAP
```

## 📊 模型性能

| 模型 | 数据集 | AP | AP@0.5 | AP@0.75 | 权重文件 |
|------|--------|----|--------|---------|----------|
| HRNet-W32 | 自定义双杠数据集 | - | - | - | [下载](releases) |

*注：具体性能指标请根据实际训练结果填写*

## 📁 项目结构

```
mmpose-21keypoints/
├── mmpose/
│   └── datasets/
│       └── datasets/
│           └── body/
│               └── coco_parallel_dataset.py  # 21点数据集类
├── configs/
│   └── _base_/
│       └── datasets/
│           └── coco_parallel.py  # 21点数据集配置
├── checkpoints/  # 权重文件目录（需自行下载）
├── tools/  # 训练和测试工具
├── demo/  # 演示脚本
└── requirements.txt
```

## 📝 数据集格式

本项目使用 COCO 格式的数据集，包含21个关键点标注。数据集格式说明：

- **图像格式**：支持常见图像格式（jpg, png等）
- **标注格式**：COCO JSON格式
- **关键点数量**：21个
- **关键点顺序**：按照上述关键点定义顺序

### 数据集准备示例

```python
# 数据集标注示例
{
    "images": [...],
    "annotations": [
        {
            "keypoints": [x1, y1, v1, x2, y2, v2, ...],  # 21个关键点，每个点3个值(x, y, visibility)
            "num_keypoints": 21,
            ...
        }
    ],
    "categories": [
        {
            "keypoints": ["nose", "left_eye", ..., "right_foot"],  # 21个关键点名称
            "skeleton": [[0, 1], [1, 2], ...]  # 骨骼连接
        }
    ]
}
```

## 🔧 配置说明

主要配置文件位于 `configs/_base_/datasets/coco_parallel.py`，包含：

- 21个关键点的定义
- 骨骼连接信息
- 关键点权重和sigma值

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目基于 [Apache 2.0 许可证](LICENSE) 开源。

本项目基于 [MMPose](https://github.com/open-mmlab/mmpose) 框架开发，遵循相同的开源协议。

## 🙏 致谢

- [MMPose](https://github.com/open-mmlab/mmpose) - OpenMMLab 姿态估计工具箱
- [OpenMMLab](https://openmmlab.com/) - 开源计算机视觉算法框架

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/yourusername/mmpose-21keypoints/issues)
- 发送邮件至：your-email@example.com

## 🔗 相关链接

- [MMPose 官方文档](https://mmpose.readthedocs.io/)
- [MMPose GitHub](https://github.com/open-mmlab/mmpose)
- [OpenMMLab](https://openmmlab.com/)

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
