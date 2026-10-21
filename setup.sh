#!/bin/bash
# ============================================================
# Run once at the start of each GPU session  bash setup.sh
# ============================================================

# ===== BEFORE RUNNING THE ANNOTATION ======
# prepare.py --all          # check/prepare clips and frames

# ===== ANNOTATION RUN ORDER ===== (note: adjust the folder path in helpers.py to change the dataset)
# round1_vid.py             # round 1 (ads vs non-ads and their intervals)
# aggregate.py              # combine results
# round2.py                 # round 2 (assess relevance and annotate food ads)

# ===== AFTER ANNOTATION IS DONE =====
# ad_similarity.py          # re-calculate screenshot similarity to detect missed ads (optional)
# export_dieticians.py      # append AI annotations to the dieticians workbook

# ===== TO PREPARE THE VALIDATION SAMPLE FOR DIETICIANS =====
# dieticians_video.py       # create validation 1 set for the dieticians (just once)
# sample_validation.py      # prepare a fresh validation set for the dieticians (excluding the ones already used in another set) 


# activate the venv with source /workspace/persistent/kidad_validation/.venv/bin/activate

# run the first round script with:
# nohup python -m screenshot_filtering.round1_vid --all > logs/all.log 2>&1 &
# to check the output live: tail -f logs/all.log

# or just one participant: 
# nohup python -m screenshot_filtering.round1_vid --enrol 23841519278 > logs/23841519278.log 2>&1 &

# aggregate the results from round 1:
# nohup python -m screenshot_filtering.aggregate > logs/aggregate.log 2>&1 &

# and then run the second round script with:
# nohup python -m screenshot_filtering.round2 > logs/round2_test.log 2>&1 &
# tail -f logs/round2_test.log



set -e  # stop on first error

mkdir -p logs

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
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"





# ============= NOTES ON FUTURE ANALYSIS ==============
# === Screenshot-level detection of relevant advertising.
# Map the AI segments and their relevance decisions back to screenshot IDs. Compare those with the manual labels across the full sample. 
# This captures relevant ads missed entirely by the AI, and avoids requiring perfect agreement about grouping.

# === Ad-group-level characterisation.
# Match human and AI groups using participant ID and overlapping screenshot IDs. Compare the detailed labels for matched groups, 
# while separately recording missed groups, extra groups, splits, and merges. Do not force ambiguous matches.