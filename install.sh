#!/bin/bash
# to run this script, first make it executable (only once): `chmod +x ./persistent/AI_validation/install2.sh`
# then run it with `./persistent/AI_validation/install2.sh`

echo "===== Setting up GPU workspace... ====="

if [[ ! -z "${CONDA_DIR}" && ! -d "${CONDA_DIR}" ]] ; then
    echo "Conda not installed, installing it."

    echo "export CONDA_DIR=$CONDA_DIR" >> ~/.zshrc
    echo "export CONDA_DIR=$CONDA_DIR" >> ~/.bashrc

    # Automatically download the latest release of mambaforge for conda/mamba
    wget -O Mambaforge-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
    chmod +x Mambaforge-Linux-x86_64.sh
    /bin/bash Mambaforge-Linux-x86_64.sh -f -b -p "${CONDA_DIR}"
    rm "Mambaforge-Linux-x86_64.sh"

    mamba config --system --set auto_update_conda false
    mamba config --system --set show_channel_urls true

    # Install Tensorflow
    mamba install -y tensorflow tensorboard
    pip install jupyter_tensorboard

else
    echo "Conda already installed."
    if ! command -v mamba &> /dev/null
    then
        echo "Mamba not installed. Installing it."
        conda install -y -c conda-forge mamba
    fi
fi



# Install required Python packages
echo "======== Installing Python packages... ========"
#conda install -y -c torch torchvision transformers openai 
pip install --upgrade setuptools pip wheel
#pip install torch torchvision ollama requests mistralai pillow pandas numpy accelerate openai qwen_vl_utils timm einops openpyxl 
python -m pip install -r requirements.txt

# install the Transformers library with the version made for Gemma 3
#pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

# Add kidad-validation to PYTHONPATH
if ! grep -q "PYTHONPATH=" ~/.bashrc; then
    echo 'export PYTHONPATH="/workspace/persistent/kidad-validation"' >> ~/.bashrc
fi
export PYTHONPATH="/workspace/persistent/kidad-validation"

# Check if GPU is available
echo "===== Checking GPU availability... ====="
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

echo "=== INSTALLATION COMPLETE !!! ==="