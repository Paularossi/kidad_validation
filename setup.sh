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
echo " Step 4: Python requirements"
echo "========================================"
pip install -r requirements.txt
echo "Python requirements installed"


echo "========================================"
echo " Done! Environment ready."
echo "========================================"
# create logs folder
mkdir -p logs
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"




# ============= NOTES ON FUTURE ANALYSIS ==============
# === Screenshot-level detection of relevant advertising.
# Map the AI segments and their relevance decisions back to screenshot IDs. Compare those with the manual labels across the full sample. 
# This captures relevant ads missed entirely by the AI, and avoids requiring perfect agreement about grouping.

# === Ad-group-level characterisation.
# Match human and AI groups using participant ID and overlapping screenshot IDs. Compare the detailed labels for matched groups, 
# while separately recording missed groups, extra groups, splits, and merges. Do not force ambiguous matches.