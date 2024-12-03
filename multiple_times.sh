#!/bin/bash

# 启用set -e，确保脚本遇到错误时退出
set -e

# 设置训练脚本路径
TRAIN_SCRIPT="/home/czh/Soli_Fusion_Mamba/train.py"

# 执行训练脚本5次
for i in {1..5}
do
  echo "开始第 $i 次训练..."
  
  # 使用python命令执行训练脚本，等待完成后才继续
  python $TRAIN_SCRIPT
  
  echo "第 $i 次训练完成。"
done

echo "所有训练已完成。"
