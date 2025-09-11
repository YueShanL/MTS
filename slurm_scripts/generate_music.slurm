#!/bin/bash
#SBATCH --job-name=musicgen-inference
#SBATCH --account=gangyiz@spartan.hpc.unimelb.edu.au   
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00            
#SBATCH --output=generate_%j.out
#SBATCH --error=generate_%j.err

# 加载模块
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.7.0

# 激活环境
source $HOME/venvs/musicgen/bin/activate
cd $HOME/MTS

echo "🎼 Starting Music Generation"
echo "Using model: outputs/best_model"

# 生成音乐
python generate_music.py \
    --model_path outputs/best_model \
    --batch_generate \
    --output_dir generated_on_spartan

echo "✅ Music generation completed!"
echo "Generated files in: generated_on_spartan/"
