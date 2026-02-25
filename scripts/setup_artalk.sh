#!/bin/bash
# Being — Phase 1: ARTalk Setup Script
# Run this on a fresh Vast.ai instance (RTX 4090 or similar)
# Requires: CUDA 12.x, ~10GB disk for models
set -e

echo "=== Phase 1: ARTalk Setup ==="

# 1. Clone ARTalk
cd /workspace
if [ ! -d "ARTalk" ]; then
    git clone --recurse-submodules https://github.com/xg-chu/ARTalk.git
    echo "ARTalk cloned."
else
    echo "ARTalk already exists."
fi
cd ARTalk

# 2. Create conda env (if conda available) or use pip
if command -v conda &> /dev/null; then
    echo "Setting up conda environment..."
    conda env create -f environment.yml -n ARTalk || conda env update -f environment.yml -n ARTalk
    eval "$(conda shell.bash hook)"
    conda activate ARTalk
else
    echo "No conda found, installing deps with pip..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install pytorch3d transformers lightning gradio scipy opencv-python trimesh
fi

# 3. Install diff-gaussian-rasterization for GAGAvatar rendering
if ! python -c "import diff_gaussian_rasterization" 2>/dev/null; then
    echo "Installing diff-gaussian-rasterization..."
    git clone --recurse-submodules https://github.com/xg-chu/diff-gaussian-rasterization.git
    pip install ./diff-gaussian-rasterization
    rm -rf ./diff-gaussian-rasterization
fi

# 4. Download model weights
echo "Downloading model weights via build_resources.sh..."
bash ./build_resources.sh

# 5. Upload test audio
echo ""
echo "=== Setup Complete ==="
echo ""
echo "To test with your audio, copy it to this machine and run:"
echo "  python inference.py -a /path/to/your_audio.wav --shape_id mesh --style_id natural_0"
echo ""
echo "For the Gradio web UI:"
echo "  python inference.py --run_app"
echo ""
echo "Output will be saved to render_results/ARTAvatar_wav2vec/"
