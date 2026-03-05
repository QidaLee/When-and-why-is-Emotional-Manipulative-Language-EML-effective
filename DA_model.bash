#!/bin/bash
#SBATCH -J DA_training
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 02:00:00

# 1. 加载基础模块
module purge
module load GCC/12.3.0 CUDA/12.1.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# 2. 如果还没有Conda环境，先创建
# 注意：这一步需要在登录节点先执行一次
# module load Conda
# conda create -p /mimer/NOBACKUP/groups/yourgroup/DA_mrda_bert python=3.11
# conda activate /mimer/NOBACKUP/groups/yourgroup/DA_mrda_bert
# pip install transformers numpy pandas scikit-learn joblib nltk tqdm

# 3. 在作业脚本中直接source激活脚本
python3 -m venv DA_mrda_bert
source DA_mrda_bert/bin/activate




# 4. 验证环境
which python
python --version
