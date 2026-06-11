#!/bin/bash
# =============================================================
# WMT26 Environment Setup Script — FINAL
# PyTorch 2.6.0 + CUDA 12.4 | Python 3.12 | RHEL 9 | Offline
# =============================================================
# USAGE:
#   On LOCAL machine (internet):     bash setup_env.sh local
#   On CLUSTER       (no internet):  bash setup_env.sh cluster
# =============================================================

set -e

PYTHON_VER="3.12"
ABI="cp312"
PLATFORM="manylinux2014_x86_64"
WHEEL_DIR="$HOME/wmt_wheelhouse"
SCRATCH="/scratch/p142503002-swapnilh"
VENV_PATH="$SCRATCH/wmt_venv"
TORCH_VER="2.6.0"
CUDA_TAG="cu124"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }

# =============================================================
# LOCAL MACHINE STEPS
# =============================================================
local_setup() {

  info "=== [0/8] Installing python3.12-venv (fixes ensurepip error) ==="
  sudo apt-get update -y
  sudo apt-get install -y python3.12-venv python3.12-dev python3-pip git

  info "=== [1/8] Creating fresh local venv ==="
  rm -rf "$HOME/wmt_local_venv"
  python3.12 -m venv "$HOME/wmt_local_venv"
  source "$HOME/wmt_local_venv/bin/activate"
  pip install --upgrade pip

  info "=== [2/8] Installing PyTorch 2.6.0 + CUDA 12.4 first ==="
  pip install torch==2.6.0 \
    --index-url "$TORCH_INDEX"

  info "=== [3/8] Installing all other packages at exact versions ==="
  pip install \
    transformers==4.44.2 \
    tokenizers==0.19.1 \
    sentencepiece==0.2.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    sacrebleu==2.4.3 \
    sacremoses==0.1.1 \
    mosestokenizer==1.2.1 \
    "regex==2024.5.15" \
    protobuf==4.25.3 \
    accelerate==0.34.2 \
    tqdm==4.66.4 \
    nltk==3.8.1 \
    "huggingface-hub==0.23.4" \
    peft==0.12.0 \
    datasets==2.21.0 \
    "sentence-transformers==3.0.1" \
    "unbabel-comet==2.2.4" \
    PyYAML==6.0.2

  info "=== [4/8] Cloning and installing Indic packages ==="
  cd "$HOME"
  git clone https://github.com/VarunGumma/IndicTransToolkit || warn "Already cloned"
  git clone https://github.com/anoopkunchukuttan/indic_nlp_library || warn "Already cloned"
  pip install --no-build-isolation "$HOME/IndicTransToolkit"
  pip install --no-build-isolation "$HOME/indic_nlp_library"

  info "=== [5/8] Freezing exact versions (excluding torch — handled separately) ==="
  pip freeze | grep -v "^torch" > "$HOME/requirements_exact.txt"
  info "Saved: $HOME/requirements_exact.txt"

  info "=== [6/8] Downloading ALL wheels for cluster (Linux x86_64, cp312) ==="
  mkdir -p "$WHEEL_DIR"

  # Download torch wheel separately from PyTorch CDN
  info "Downloading PyTorch ${TORCH_VER}+${CUDA_TAG} wheel..."
  pip download "torch==${TORCH_VER}" \
    --index-url "$TORCH_INDEX" \
    -d "$WHEEL_DIR" \
    --no-deps

  # Download all other packages (binary wheels for RHEL9 x86_64)
  pip download \
    -r "$HOME/requirements_exact.txt" \
    -d "$WHEEL_DIR" \
    --platform "$PLATFORM" \
    --python-version "$PYTHON_VER" \
    --implementation cp \
    --abi "$ABI" \
    --only-binary=:all: || warn "Some packages lack binary wheels — running source fallback..."

  info "=== [6b/8] Source fallback for packages without binary wheels ==="
  for pkg in mosestokenizer sacremoses; do
    pip download "$pkg" -d "$WHEEL_DIR" --no-binary "$pkg" 2>/dev/null \
      || warn "Skipping source fallback for $pkg"
  done

  info "=== [7/8] Packing Indic source repos ==="
  tar -czvf "$WHEEL_DIR/indic_src.tar.gz" \
    "$HOME/IndicTransToolkit" "$HOME/indic_nlp_library"

  info "=== [8/8] Compressing everything into one archive ==="
  tar -czvf "$HOME/wmt_wheelhouse.tar.gz" -C "$HOME" wmt_wheelhouse

  echo ""
  info "======================================================"
  info " LOCAL SETUP COMPLETE"
  info "======================================================"
  info "Now transfer to cluster with:"
  echo ""
  echo "  scp -P 49122 ~/wmt_wheelhouse.tar.gz ~/requirements_exact.txt \\"
  echo "    p142503002-swapnilh@madhava.iitpkd.ac.in:$SCRATCH/"
  echo ""
  info "Then SSH into cluster and run:  bash setup_env.sh cluster"
}

# =============================================================
# CLUSTER STEPS
# =============================================================
cluster_setup() {

  info "=== [1/7] Loading Python 3.12.8 ==="
  module purge
  module load python-3.12.8
  python3 --version

  info "=== [2/7] Extracting wheel archive ==="
  cd "$SCRATCH"
  tar -xzvf wmt_wheelhouse.tar.gz
  tar -xzvf wmt_wheelhouse/indic_src.tar.gz -C "$SCRATCH"

  info "=== [3/7] Creating venv in /scratch (no quota issues) ==="
  rm -rf "$VENV_PATH"
  python3 -m venv "$VENV_PATH"
  source "$VENV_PATH/bin/activate"
  pip install --upgrade pip

  info "=== [4/7] Installing PyTorch ${TORCH_VER}+${CUDA_TAG} OFFLINE ==="
  pip install \
    --no-index \
    --find-links="$SCRATCH/wmt_wheelhouse" \
    "torch==${TORCH_VER}"

  info "=== Verifying CUDA ==="
  python3 -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"

  info "=== [5/7] Installing all other packages OFFLINE ==="
  pip install \
    --no-index \
    --find-links="$SCRATCH/wmt_wheelhouse" \
    -r "$SCRATCH/requirements_exact.txt"

  info "=== [6/7] Installing Indic packages ==="
  pip install --no-build-isolation "$SCRATCH/IndicTransToolkit"
  pip install --no-build-isolation "$SCRATCH/indic_nlp_library"

  info "=== [7/7] Verifying exact match ==="
  pip freeze > "$SCRATCH/cluster_installed.txt"
  diff <(grep -v "^torch" "$SCRATCH/requirements_exact.txt") \
       <(pip freeze | grep -v "^torch") && \
    info "PERFECT MATCH — environment is an exact replica!" || \
    warn "Minor diff found — check $SCRATCH/cluster_installed.txt"

  echo ""
  info "======================================================"
  info " CLUSTER SETUP COMPLETE"
  info "======================================================"
  info "Activate anytime with:"
  echo "  module purge && module load python-3.12.8"
  echo "  source $VENV_PATH/bin/activate"
}

# =============================================================
# ENTRY POINT
# =============================================================
case "${1:-}" in
  local)   local_setup   ;;
  cluster) cluster_setup ;;
  *)
    echo ""
    echo "Usage:"
    echo "  bash setup_env.sh local    -- run on LOCAL machine (needs internet)"
    echo "  bash setup_env.sh cluster  -- run on CLUSTER (no internet needed)"
    echo ""
    ;;
esac
