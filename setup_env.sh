#!/bin/bash
# Setup script for telluric correction pipeline on Compute Canada clusters
# Usage: source setup_env.sh

# Set library path for numpy compatibility (required on Compute Canada / CVMFS)
export LD_LIBRARY_PATH="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64:$LD_LIBRARY_PATH"

# Test numpy import
python -c "import numpy; print('NumPy importé avec succès, version:', numpy.__version__)" 2>/dev/null \
    && echo "✓ NumPy fonctionne correctement" \
    || echo "✗ Problème avec NumPy persistant"

echo "Environnement configuré pour la correction tellurique."