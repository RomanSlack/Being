#!/bin/bash
# Download pre-trained InsTaG checkpoints from Google Drive
# These are needed for few-shot adaptation (skip expensive pre-training)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/output/pretrained"

echo "━━━ Being — Download Pre-trained Checkpoints ━━━"
echo ""
echo "InsTaG provides 4 pre-trained variants:"
echo "  1. DeepSpeech audio extractor"
echo "  2. wav2vec (esperanto) audio extractor"
echo "  3. HuBERT audio extractor"
echo "  4. AVE (SyncTalk) audio extractor"
echo ""
echo "Google Drive folder:"
echo "  https://drive.google.com/drive/folders/1R77F6YN1QUldjqAi3fsXs2N8rrsRYMPP"
echo ""

# Check if gdown is available
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown for Google Drive downloads..."
    pip install gdown
fi

mkdir -p "$OUTPUT_DIR"

echo "Downloading checkpoints... (this may take a while)"
echo "If this fails, download manually from the Google Drive link above"
echo "and extract to: $OUTPUT_DIR/"
echo ""

# gdown can download Google Drive folders
gdown --folder "https://drive.google.com/drive/folders/1R77F6YN1QUldjqAi3fsXs2N8rrsRYMPP" \
    -O "$OUTPUT_DIR" --remaining-ok 2>&1 || {
    echo ""
    echo "Automatic download failed. Please download manually:"
    echo "  1. Go to: https://drive.google.com/drive/folders/1R77F6YN1QUldjqAi3fsXs2N8rrsRYMPP"
    echo "  2. Download all files"
    echo "  3. Extract to: $OUTPUT_DIR/"
    exit 1
}

echo ""
echo "✓ Checkpoints downloaded to: $OUTPUT_DIR/"
echo ""
echo "To see available checkpoints:"
ls -la "$OUTPUT_DIR/"
