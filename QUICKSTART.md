# 21点模型快速开始指南

## 🚀 5分钟快速上手

### 1. 安装（3分钟）

```bash
# 克隆仓库
git clone https://github.com/yourusername/mmpose-21keypoints.git
cd mmpose-21keypoints

# 安装依赖
pip install -r requirements.txt
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install -e .
```

### 2. 下载权重（1分钟）

1. 从网盘下载：`best_coco_AP_epoch_110.pth`
2. 放到 `checkpoints/` 目录

```bash
mkdir -p checkpoints
# 将下载的文件复制到 checkpoints/
cp /path/to/best_coco_AP_epoch_110.pth checkpoints/
```

### 3. 开始推理（1分钟）

```bash
# 使用示例脚本
python inference_example.py your_image.jpg

# 或使用 Python
python -c "
from mmpose.apis import MMPoseInferencer
inferencer = MMPoseInferencer(
    pose2d='configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel.py',
    pose2d_weights='checkpoints/best_coco_AP_epoch_110.pth'
)
inferencer('your_image.jpg', vis_out_dir='results')
"
```

## ✅ 验证安装

```bash
python -c "from mmpose.datasets import CocoParallelDataset; print('✓ 安装成功')"
```

## 📖 详细文档

查看 [21点模型使用说明.md](./21点模型使用说明.md) 获取完整文档。
