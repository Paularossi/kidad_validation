#!/bin/bash
# ============================================================
# Run once at the start of each GPU session  bash setup.sh
# ============================================================

# ===== run order:
# python round1.py --all        # round 1 (ads vs non-ads)
# python aggregate.py           # combine results
# python round2.py              # round 2 (food ads annotation)

# activate the venv with source /workspace/persistent/.venv/bin/activate

# then run the first round script with:
# nohup python -m round1_vid --enrol 23841517978 > logs/23841517978.log 2>&1 &
# to check the output live: tail -f logs/23841517978.log

# or all participants
# nohup python -m round1 --all > logs/all_28_38.log 2>&1 &

# and second round script with:
# nohup python -m round2 > logs/round2_test.log 2>&1 &
# tail -f logs/round2_test.log


set -e  # stop on first error

echo "========================================"
echo " Step 1: System dependencies"
echo "========================================"
apt-get update -qq
apt-get install -y ffmpeg libgl1
echo "System dependencies installed"


echo "========================================"
echo " Step 2: Create and activate virtual environment"
echo "========================================"
if [ ! -d "/workspace/persistent/.venv" ]; then
    python3 -m venv /workspace/persistent/.venv
    echo "Virtualenv created at /workspace/persistent/.venv"
else
    echo "Virtualenv already exists, skipping creation"
fi
source /workspace/persistent/.venv/bin/activate
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
pip install -r requirements.txt
echo "Python requirements installed"


echo "========================================"
echo " Done! Environment ready."
echo "========================================"
# create logs folder
mkdir -p logs
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"



# to log into the OC client
# copy the login command from the DSRI web UI
# then get the pod ID `oc get pod --selector app=kidad-gpu-jup | xargs -I{} oc cp <folder_to_copy> {}:<absolute_path_in_pod>`: kidad-gpu-jup-6c4f6bc8ff-2hm9j
# copy images from local to the pod: 
# cd "C:/Users/P70090005/OneDrive - Maastricht University/Documents/kidad_validation/data/food_ads"
# `oc cp 23841529678 kidad-gpu-jup-6c4f6bc8ff-dfscj:/workspace/persistent/kidad/data/participants/23841529678`

# copy from pod to local:
# `oc cp kidad-gpu-jup-6c4f6bc8ff-dfscj:/workspace/persistent/kidad/data/results/ results/`