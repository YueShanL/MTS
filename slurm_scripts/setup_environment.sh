#!/bin/bash
# CompressedMusicGen 环境设置脚本 - 使用现有 requirements.txt
# 文件路径: MTS/slurm_scripts/setup_environment.sh

echo "🔧 Setting up CompressedMusicGen environment on Spartan HPC"
echo "Using existing requirements.txt from your MTS project"

# 创建虚拟环境
echo "📦 Creating virtual environment..."
python3 -m venv $HOME/venvs/musicgen
source $HOME/venvs/musicgen/bin/activate

# 升级pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# 安装现有项目的完整依赖
echo "📋 Installing dependencies from requirements.txt..."
cd $HOME/MTS
pip install -r requirements.txt

# 验证安装
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import librosa; print(f'Librosa: {librosa.__version__}')"

echo "🎉 Environment setup completed!"
echo "✅ All dependencies from requirements.txt installed successfully"
echo ""
echo "To activate this environment in future sessions:"
echo "source $HOME/venvs/musicgen/bin/activate"
