#!/bin/bash
# Setup script for telluric correction pipeline on Compute Canada clusters
# Usage: source setup_env.sh
#
# If numpy fails with "libcpupower.so.0 not found", reinstall it:
#   ~/Venv/tellu_env/bin/pip install --force-reinstall numpy
# Do NOT set LD_LIBRARY_PATH to CVMFS paths — this breaks libc and segfaults the shell.

# Test numpy import
python -c "import numpy; print('NumPy importé avec succès, version:', numpy.__version__)" 2>/dev/null \
    && echo "✓ NumPy fonctionne correctement" \
    || echo "✗ Problème avec NumPy — voir commentaire ci-dessus"

echo "Environnement configuré pour la correction tellurique."
python -c "import numpy; print('NumPy importé avec succès, version:', numpy.__version__)" 2>/dev/null \
    && echo "✓ NumPy fonctionne correctement" \
    || echo "✗ Problème avec NumPy persistant"

echo "Environnement configuré pour la correction tellurique."