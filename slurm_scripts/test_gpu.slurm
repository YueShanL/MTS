#!/bin/bash
#SBATCH --job-name=gpu-check
#SBATCH --account=gangyiz@spartan.hpc.unimelb.edu.au  
#SBATCH --partition=interactive    
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/test_gpu_%j.out
#SBATCH --error=logs/test_gpu_%j.err

echo "🔍 Checking assigned GPU on $(hostname)"
module purge
module load CUDA/11.7.0
nvidia-smi
EOF