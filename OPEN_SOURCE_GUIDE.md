# 21点骨骼模型开源指南

## 项目概述

本项目基于 MMPose 框架，扩展了标准的 COCO 17点关键点模型，新增了4个脚部关键点（左右脚跟和左右脚尖），形成了21点骨骼模型。该模型专门针对双杠动作姿态估计进行了优化和训练。

## 开源准备清单

### 1. 代码整理

#### 1.1 核心修改文件
确保以下文件已正确修改并包含在仓库中：

- **数据集定义**：
  - `mmpose/datasets/datasets/body/coco_parallel_dataset.py` - 21点数据集类
  - `configs/_base_/datasets/coco_parallel.py` - 数据集配置文件

- **注册文件**：
  - `mmpose/datasets/datasets/body/__init__.py` - 确保已注册 `CocoParallelDataset`

#### 1.2 清理不必要的文件
- 删除临时文件、日志文件
- 确保 `.gitignore` 正确配置（已包含检查点文件）
- 删除个人路径硬编码（如 `/home/satuo/...`）

### 2. 权重文件处理

#### 2.1 权重文件位置
训练好的权重文件：`checkpoints/best_coco_AP_epoch_110.pth`

#### 2.2 权重文件上传方案

**方案A：GitHub Releases（推荐）**
- 优点：版本管理清晰，下载方便，不占用仓库空间
- 步骤：
  1. 在 GitHub 仓库创建 Release
  2. 上传 `.pth` 文件作为附件
  3. 在 README 中提供下载链接

**方案B：云存储服务**
- 使用 Google Drive、百度网盘、阿里云OSS等
- 在 README 中提供下载链接和提取码（如需要）

**方案C：Git LFS（大文件支持）**
- 如果文件 < 100MB，可以直接提交
- 如果文件较大，使用 Git LFS：
  ```bash
  git lfs install
  git lfs track "*.pth"
  git add .gitattributes
  git add checkpoints/best_coco_AP_epoch_110.pth
  ```

### 3. 文档编写

#### 3.1 README.md 结构建议

```markdown
# MMPose 21点骨骼模型

## 简介
本项目基于 MMPose 框架，扩展了 COCO 17点关键点模型，新增4个脚部关键点，形成21点骨骼模型。

## 关键点定义
0-16: COCO标准17点
17: left_heel (左脚跟)
18: right_heel (右脚跟)
19: left_foot (左脚尖)
20: right_foot (右脚尖)

## 安装

### 环境要求
- Python >= 3.7
- PyTorch >= 1.8
- CUDA >= 10.2 (如使用GPU)

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/mmpose-21keypoints.git
cd mmpose-21keypoints
```

2. 安装依赖
```bash
pip install -r requirements.txt
pip install -e .  # 以开发模式安装
```

3. 下载权重文件
```bash
# 从 GitHub Releases 下载
wget https://github.com/yourusername/mmpose-21keypoints/releases/download/v1.0/best_coco_AP_epoch_110.pth
mkdir -p checkpoints
mv best_coco_AP_epoch_110.pth checkpoints/
```

## 使用方法

### 推理示例
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

### 训练
```bash
python tools/train.py configs/body_2d_keypoint/your_config.py
```

## 模型性能
- 训练数据集：自定义双杠动作数据集
- 评估指标：COCO AP
- 最佳模型：epoch 110

## 数据集格式
数据集采用 COCO 格式，包含21个关键点标注。

## 许可证
本项目基于 Apache 2.0 许可证开源。

## 致谢
- 基于 [MMPose](https://github.com/open-mmlab/mmpose) 框架
- 参考 COCO 数据集格式
```

#### 3.2 创建使用示例
创建 `examples/` 目录，包含：
- `inference_example.py` - 推理示例
- `training_example.py` - 训练示例
- `visualization_example.py` - 可视化示例

### 4. GitHub 仓库设置

#### 4.1 创建仓库
1. 在 GitHub 创建新仓库（建议名称：`mmpose-21keypoints` 或 `parallel-bar-pose-estimation`）
2. 添加仓库描述和标签

#### 4.2 初始化 Git 仓库
```bash
cd /home/satuo/code/Train_Parallel_Model
git init
git add .
git commit -m "Initial commit: MMPose 21-keypoint model"
git branch -M main
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main
```

#### 4.3 创建 Release
1. 在 GitHub 仓库页面，点击 "Releases" -> "Create a new release"
2. 填写版本号（如 v1.0.0）
3. 上传权重文件 `best_coco_AP_epoch_110.pth`
4. 添加 Release 说明

### 5. 代码清理检查

#### 5.1 检查硬编码路径
搜索并替换所有硬编码路径：
```bash
grep -r "/home/satuo" .
grep -r "absolute/path" .
```

#### 5.2 检查配置文件
确保配置文件中没有硬编码路径，使用相对路径或环境变量。

#### 5.3 添加必要的注释
- 在关键修改处添加注释说明
- 说明21点模型与标准COCO模型的区别

### 6. 许可证和版权

#### 6.1 检查许可证
- MMPose 使用 Apache 2.0 许可证
- 确保你的修改也遵循相同的许可证
- 在 README 中明确说明

#### 6.2 添加致谢
在 README 中感谢 MMPose 团队和 OpenMMLab。

### 7. 测试验证

#### 7.1 创建测试脚本
创建简单的测试脚本，验证：
- 模型可以正确加载
- 推理可以正常运行
- 关键点输出格式正确

#### 7.2 在干净环境中测试
在全新的环境中测试安装和使用流程，确保其他人可以复现。

## 推荐的发布流程

### 第一步：代码整理
1. ✅ 清理临时文件和日志
2. ✅ 检查并修复硬编码路径
3. ✅ 添加必要的注释和文档
4. ✅ 确保所有核心文件都在仓库中

### 第二步：文档编写
1. ✅ 编写详细的 README.md
2. ✅ 创建使用示例
3. ✅ 添加模型性能说明
4. ✅ 添加数据集格式说明

### 第三步：GitHub 设置
1. ✅ 创建 GitHub 仓库
2. ✅ 初始化并推送代码
3. ✅ 创建第一个 Release
4. ✅ 上传权重文件

### 第四步：验证和优化
1. ✅ 在干净环境中测试安装
2. ✅ 根据反馈优化文档
3. ✅ 添加更多示例和教程

## 常见问题

### Q: 权重文件太大怎么办？
A: 使用 GitHub Releases 或云存储服务，不要直接提交到仓库。

### Q: 如何让其他人更容易使用？
A: 
- 提供详细的安装说明
- 创建简单的使用示例
- 提供预训练权重下载链接
- 添加常见问题解答

### Q: 需要提供训练数据吗？
A: 通常不需要，但可以：
- 提供数据集的格式说明
- 提供数据标注工具的使用指南
- 如果数据集可以公开，可以单独发布

### Q: 如何维护项目？
A:
- 及时回复 Issues
- 定期更新文档
- 修复发现的 Bug
- 考虑添加新功能

## 总结

开源一个模型项目需要：
1. **代码质量**：清晰、可读、有注释
2. **文档完善**：README、示例、API文档
3. **易于使用**：简单的安装和使用流程
4. **资源可获取**：权重文件易于下载
5. **持续维护**：及时回复问题，更新文档

按照以上步骤，你的21点骨骼模型就可以成功开源了！
