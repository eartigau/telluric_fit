#!/bin/bash
# Setup script for telluric correction pipeline on Compute Canada clusters
# Usage: source setup_env.sh
#
# One-time setup: create a symlink so numpy finds libcpupower.so.0 without
# polluting LD_LIBRARY_PATH (which breaks libc symbol resolution):
#   ln -s /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64/libcpupower.so.0 \
#         ~/Venv/tellu_env/lib/libcpupower.so.0

# Test numpy import
python -c "import numpy; print('NumPy importé avec succès, version:', numpy.__version__)" 2>/dev/null \
    && echo "✓ NumPy fonctionne correctement" \
    || echo "✗ Problème avec NumPy persistant"

echo "Environnement configuré pour la correction tellurique."