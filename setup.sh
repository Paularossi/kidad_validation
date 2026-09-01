#!/bin/bash
# ============================================================
# Run once at the start of each GPU session  bash setup.sh
# ============================================================

# ===== run order:
# round1_vid                # round 1 (ads vs non-ads)
# aggregate.py              # combine results
# rerun_failed_clips.py     # re-run failed clips from round 1
# round2.py                 # round 2 (food ads annotation)

# activate the venv with source /workspace/persistent/.venv/bin/activate

# then run the first round script with:
# nohup python -m screenshot_filtering.round1_vid --enrol 23841517978 > logs/23841517978.log 2>&1 &
# to check the output live: tail -f logs/23841517978.log

# or all participants
# nohup python -m screenshot_filtering.round1_vid --all > logs/all_1_17.log 2>&1 &

# all failed clips
# nohup python -m screenshot_filtering.rerun_failed_clips > logs/rerun_failed.log 2>&1 &

# and second round script with:
# nohup python -m screenshot_filtering.round2 > logs/round2_test.log 2>&1 &
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
# cd "C:/Users/P70090005/OneDrive - Maastricht University/Documents/kidad_validation/data/participants"
# oc cp 23841526078 kidad-gpu-jup-5f87fb7559-mlk4l:/workspace/persistent/kidad/data/participants/23841526078

# copy from pod to local:
# oc cp kidad-gpu-jup-5f87fb7559-mlk4l:/workspace/persistent/kidad/data/results_videos/ results_videos/

# to delete folders:
# oc exec kidad-gpu-jup-6c4f6bc8ff-xrsqt -- rm -rf /workspace/persistent/kidad/data/participants/

# to check the storage:
# oc exec kidad-gpu-jup-6c4f6bc8ff-xrsqt -- df -h /workspace/persistent/

