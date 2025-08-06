#!/bin/bash
echo "当前Python路径: $(which python)"
echo "当前环境: $CONDA_DEFAULT_ENV"

# 安装verl依赖
echo "安装verl..."
pip install -e ./verl

echo "尝试安装预编译的flash-attn..."
pip install flash-attn==2.8.0.post2 --no-build-isolation

# 安装verl依赖
echo "安装verl..."
pip install -e ./verl


# 安装当前项目
echo "安装当前项目..."
# pip install -e . --force-reinstall 2>&1 | grep -v "Cannot uninstall blinker" || true


# 切换到deepcoder示例目录
echo "切换到deepcoder目录..."
cd examples/deepcoder

# 准备deepcoder数据
echo "准备deepcoder数据..."
python prepare_deepcoder_data.py

echo "所有任务完成！"