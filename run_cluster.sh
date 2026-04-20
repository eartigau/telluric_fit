#!/bin/bash
# Wrapper script for running telluric correction pipeline on Compute Canada clusters
# This script automatically sets up the environment before running the pipeline

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Configuration environnement cluster ===${NC}"

# Set library path for numpy compatibility
export LD_LIBRARY_PATH="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64:$LD_LIBRARY_PATH"
echo -e "${GREEN}✓${NC} LD_LIBRARY_PATH configuré"

# Check if we're in the right conda environment
if [[ "$CONDA_DEFAULT_ENV" != "tellu_env" ]]; then
    echo -e "${YELLOW}⮞${NC} Activation de l'environnement tellu_env..."
    conda activate tellu_env
fi

# Test numpy import
echo -e "${YELLOW}⮞${NC} Test d'importation NumPy..."
if python -c "import numpy; print('NumPy version:', numpy.__version__)" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} NumPy fonctionne correctement"
else
    echo -e "${RED}✗${NC} Problème avec NumPy persistant"
    echo -e "${YELLOW}⮞${NC} Essayez de réinstaller numpy: conda install -c conda-forge numpy"
    exit 1
fi

echo -e "${YELLOW}=== Lancement du pipeline tellurique ===${NC}"

# Run the pipeline with all provided arguments
python run_pipeline.py "$@"