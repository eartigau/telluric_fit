#!/usr/bin/env bash
# Install third-party dependencies for telluric_fit
# Designed for Compute Canada (Narval/FIR) with CVMFS wheelhouse.
#
# Usage:
#   module load python/3.12.4   # or whichever python/3.12.x is available
#   ./install_deps.sh
#   source ~/telluric_env/bin/activate

VENV_DIR="${HOME}/telluric_env"

echo "Creating virtualenv at ${VENV_DIR} ..."
# --no-download: use only already-present pip/setuptools (no network), avoids broken /tmp
virtualenv --no-download "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"

# Upgrade pip using only the CVMFS wheelhouse (--no-index = no network)
pip install --no-index --upgrade pip

# Install all dependencies from the CVMFS wheelhouse
pip install --no-index \
    numpy \
    matplotlib \
    astropy \
    scipy \
    tqdm \
    numexpr \
    PyYAML

echo ""
echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
