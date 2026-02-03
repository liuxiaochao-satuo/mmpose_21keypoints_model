#!/bin/bash
# 开源准备脚本
# 用于检查和准备项目以进行开源

echo "=========================================="
echo "MMPose 21点模型开源准备脚本"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查硬编码路径
echo -e "\n${YELLOW}[1/5] 检查硬编码路径...${NC}"
HARDCODED_PATHS=$(grep -r "/home/satuo" . --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=.pytest_cache 2>/dev/null | grep -v ".git" | head -10)
if [ -z "$HARDCODED_PATHS" ]; then
    echo -e "${GREEN}✓ 未发现硬编码路径${NC}"
else
    echo -e "${RED}✗ 发现硬编码路径:${NC}"
    echo "$HARDCODED_PATHS"
    echo -e "${YELLOW}请手动检查并修复这些路径${NC}"
fi

# 检查权重文件
echo -e "\n${YELLOW}[2/5] 检查权重文件...${NC}"
if [ -f "checkpoints/best_coco_AP_epoch_110.pth" ]; then
    FILE_SIZE=$(du -h checkpoints/best_coco_AP_epoch_110.pth | cut -f1)
    echo -e "${GREEN}✓ 找到权重文件: checkpoints/best_coco_AP_epoch_110.pth (大小: $FILE_SIZE)${NC}"
    echo -e "${YELLOW}提示: 权重文件应该通过 GitHub Releases 或云存储提供，不要直接提交到仓库${NC}"
else
    echo -e "${YELLOW}⚠ 未找到权重文件，请确保在 Releases 中提供${NC}"
fi

# 检查核心文件
echo -e "\n${YELLOW}[3/5] 检查核心文件...${NC}"
CORE_FILES=(
    "mmpose/datasets/datasets/body/coco_parallel_dataset.py"
    "configs/_base_/datasets/coco_parallel.py"
    "mmpose/datasets/datasets/body/__init__.py"
    "README.md"
    "requirements.txt"
)

ALL_EXIST=true
for file in "${CORE_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ 缺失: $file${NC}"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = true ]; then
    echo -e "${GREEN}所有核心文件都存在${NC}"
else
    echo -e "${RED}请确保所有核心文件都存在${NC}"
fi

# 检查 .gitignore
echo -e "\n${YELLOW}[4/5] 检查 .gitignore...${NC}"
if [ -f ".gitignore" ]; then
    if grep -q "\.pth" .gitignore && grep -q "checkpoints/" .gitignore; then
        echo -e "${GREEN}✓ .gitignore 已正确配置${NC}"
    else
        echo -e "${YELLOW}⚠ .gitignore 可能未包含权重文件规则${NC}"
    fi
else
    echo -e "${RED}✗ 未找到 .gitignore 文件${NC}"
fi

# 检查文档
echo -e "\n${YELLOW}[5/5] 检查文档...${NC}"
DOC_FILES=(
    "README.md"
    "OPEN_SOURCE_GUIDE.md"
)

for file in "${DOC_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${YELLOW}⚠ 建议创建: $file${NC}"
    fi
done

# 总结
echo -e "\n=========================================="
echo -e "${GREEN}检查完成！${NC}"
echo -e "=========================================="
echo -e "\n下一步操作："
echo -e "1. 修复发现的硬编码路径"
echo -e "2. 准备权重文件上传到 GitHub Releases"
echo -e "3. 完善 README.md 文档"
echo -e "4. 初始化 Git 仓库并推送代码"
echo -e "5. 创建第一个 Release 并上传权重文件"
echo -e "\n详细说明请参考: OPEN_SOURCE_GUIDE.md"
