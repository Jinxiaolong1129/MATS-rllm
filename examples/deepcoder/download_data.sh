#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --mail-type=all
#SBATCH --mail-user=jinxiaolong1129@gmail.com
#SBATCH --output=slurm_mats_setup_out.txt
#SBATCH --error=slurm_mats_setup_error.txt

# 切换到工作目录
cd /data/xuandong_zhao/mnt/xiaolong/MATS-rllm

# 设置环境变量
set -x
export PYTHONUNBUFFERED=1

# 初始化conda (fix for conda init error)
echo "初始化conda..."
eval "$(/data/xuandong_zhao/anaconda3/bin/conda shell.bash hook)"

# 激活conda环境
echo "激活rllm环境..."
conda activate rllm

# 验证环境激活
echo "当前Python路径: $(which python)"
echo "当前环境: $CONDA_DEFAULT_ENV"

echo "开始安装依赖..."

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
pip install -e . --force-reinstall 2>&1 | grep -v "Cannot uninstall blinker" || true

python -c "import rllm; print('✓ rllm 导入成功')"

# 切换到deepcoder示例目录
echo "切换到deepcoder目录..."
cd examples/deepcoder

# 准备deepcoder数据
echo "准备deepcoder数据..."
python prepare_deepcoder_data.py

echo "所有任务完成！"