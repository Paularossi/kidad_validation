#!/bin/bash
# ============================================================
# Run once at the start of each GPU session  bash kidad/setup.sh
# ============================================================

# activate the venv with source /root/.venv/bin/activate
# then run the main script with `nohup python -m kidad.screenshot_filtering.example &`

set -e  # stop on first error

echo "========================================"
echo " Step 1: System dependencies"
echo "========================================"
apt-get update -qq
apt-get install -y ffmpeg libgl1
echo "System dependencies installed"


echo "========================================"
echo " Step 2: Activate virtual environment"
echo "========================================"
source /root/.venv/bin/activate
echo "Virtualenv active: $(which python)"


echo "========================================"
echo " Step 3: PyTorch (CUDA 12.8)"
echo "========================================"
pip install \
    "torch==2.10.0" \
    "torchvision==0.25.0" \
    --index-url https://download.pytorch.org/whl/cu128
echo "PyTorch installed"


echo "========================================"
echo " Step 4: Flash Attention"
echo "========================================"
pip install \
    "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
echo "Flash Attention installed"


echo "========================================"
echo " Step 6: Python requirements"
echo "========================================"
pip install -r kidad/requirements.txt
echo "Python requirements installed"


echo "========================================"
echo " Done! Environment ready."
echo "========================================"
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"