#!/bin/bash
#SBATCH --job-name=compressed-musicgen
#SBATCH --account=gangyiz@spartan.hpc.unimelb.edu.au     
#SBATCH --partition=gpu-a100      
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               
#SBATCH --mem=32G                 
#SBATCH --time=24:00:00            
#SBATCH --output=musicgen_%j.out
#SBATCH --error=musicgen_%j.err

# 加载模块
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.7.0

# 激活虚拟环境
source $HOME/venvs/musicgen/bin/activate

# 设置工作目录
cd $HOME/MTS

# 运行训练
echo "🎵 Starting CompressedMusicGen Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python training/train_complete.py \
    --epochs 50 \
    --batch_size 8 \
    --output_dir outputs \
    --device cuda

echo "✅ Training completed!"
