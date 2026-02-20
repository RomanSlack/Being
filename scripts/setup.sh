#!/bin/bash
# Being — Automated Setup Script
# Handles: InsTaG clone, conda env, CUDA extensions, external model downloads
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXTERN_DIR="$PROJECT_ROOT/extern"
INSTAG_DIR="$EXTERN_DIR/InsTaG"

echo "════════════════════════════════════════════════════"
echo "  Being — Setup"
echo "════════════════════════════════════════════════════"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# ─────────────────────────────────────────────────────
# 1. Clone InsTaG submodule
# ─────────────────────────────────────────────────────
echo "━━━ [1/7] InsTaG Repository ━━━"
if [ -f "$INSTAG_DIR/synthesize_fuse.py" ]; then
    echo "✓ InsTaG already cloned"
else
    echo "Cloning InsTaG..."
    mkdir -p "$EXTERN_DIR"
    git clone --recursive https://github.com/Fictionarry/InsTaG.git "$INSTAG_DIR"
    cd "$INSTAG_DIR" && git submodule update --init --recursive && cd "$PROJECT_ROOT"
    echo "✓ InsTaG cloned"
fi

# ─────────────────────────────────────────────────────
# 2. Conda environment
# ─────────────────────────────────────────────────────
echo ""
echo "━━━ [2/7] Conda Environment ━━━"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "✗ conda not found. Install Miniconda or Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create/update the environment
if conda env list | grep -q "^being "; then
    echo "✓ 'being' conda env already exists"
    echo "  To update: conda env update -f environment.yml"
else
    echo "Creating 'being' conda env..."
    conda env create -f "$PROJECT_ROOT/environment.yml"
    echo "✓ Conda env created"
fi

echo ""
echo "Activating being environment..."
eval "$(conda shell.bash hook)"
conda activate being

# ─────────────────────────────────────────────────────
# 3. Install Being package
# ─────────────────────────────────────────────────────
echo ""
echo "━━━ [3/7] Being Package ━━━"
pip install -e "$PROJECT_ROOT" 2>&1 | tail -5
echo "✓ Being package installed"

# ─────────────────────────────────────────────────────
# 4. PyTorch3D
# ─────────────────────────────────────────────────────
echo ""
echo "━━━ [4/7] PyTorch3D ━━━"
if python -c "import pytorch3d" 2>/dev/null; then
    echo "✓ PyTorch3D already installed"
else
    echo "Installing PyTorch3D (this may take a few minutes)..."
    pip install "git+https://github.com/facebookresearch/pytorch3d.git" 2>&1 | tail -5
    echo "✓ PyTorch3D installed"
fi

# ─────────────────────────────────────────────────────
# 5. InsTaG pre-trained weights
# ─────────────────────────────────────────────────────
echo ""
echo "━━━ [5/7] Pre-trained Weights ━━━"
cd "$INSTAG_DIR"
if [ -f "data_utils/face_parsing/79999_iter.pth" ]; then
    echo "✓ Face parsing weights already downloaded"
else
    echo "Downloading face parsing + landmark weights..."
    bash scripts/prepare.sh 2>&1 | tail -5
    echo "✓ Weights downloaded"
fi

# ─────────────────────────────────────────────────────
# 6. EasyPortrait
# ─────────────────────────────────────────────────────
echo ""
echo "━━━ [6/7] EasyPortrait ━━━"
EASYPORTRAIT_MODEL="$INSTAG_DIR/data_utils/easyportrait/fpn-fp-512.pth"
if [ -f "$EASYPORTRAIT_MODEL" ]; then
    echo "✓ EasyPortrait model already downloaded"
else
    echo "Installing EasyPortrait dependencies..."
    pip install -U openmim 2>&1 | tail -3
    mim install mmcv-full==1.7.1 prettytable 2>&1 | tail -3

    echo "Downloading EasyPortrait model..."
    wget -q --show-progress \
        "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth" \
        -O "$EASYPORTRAIT_MODEL"
    echo "✓ EasyPortrait ready"
fi

cd "$PROJECT_ROOT"

# ─────────────────────────────────────────────────────
# 7. Basel Face Model check
# ─────────────────────────────────────────────────────
echo ""
echo "━━━ [7/7] Basel Face Model ━━━"
BFM_DIR="$INSTAG_DIR/data_utils/face_tracking/3DMM"
if [ -f "$BFM_DIR/01_MorphableModel.mat" ] || [ -f "$BFM_DIR/BFM09_model_info.mat" ]; then
    echo "✓ Basel Face Model found"
else
    echo "✗ Basel Face Model NOT found"
    echo ""
    echo "  You need to manually download it:"
    echo "  1. Go to: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details"
    echo "  2. Register (free) and download 01_MorphableModel.mat"
    echo "  3. Copy to: $BFM_DIR/"
    echo "  4. Then run:"
    echo "     cd $INSTAG_DIR/data_utils/face_tracking && python convert_BFM.py"
    echo ""
fi

# ─────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  Setup Complete"
echo "════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. conda activate being"
echo "  2. Download Basel Face Model (see above)"
echo "  3. (Optional) Install OpenFace for blink detection"
echo "  4. Download pre-trained InsTaG checkpoints:"
echo "     https://drive.google.com/drive/folders/1R77F6YN1QUldjqAi3fsXs2N8rrsRYMPP"
echo "     Extract to: $PROJECT_ROOT/output/pretrained/"
echo ""
echo "  5. Test with: being check"
echo "  6. Prepare a video: being prepare <video.mp4>"
echo ""
