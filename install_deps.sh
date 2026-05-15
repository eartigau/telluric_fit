#!/usr/bin/env bash
# Install third-party dependencies for telluric_fit
# Designed for Compute Canada (Narval/FIR) with CVMFS wheelhouse.
#
# Usage:
#   module load python/3.12.4   # or whichever python/3.12.x is available
#   ./install_deps.sh
#   source ~/telluric_env/bin/activate

set -e   # abort immediately if any command fails

VENV_DIR="${HOME}/telluric_env"

# Wipe corrupted virtualenv seed cache (causes "no .dist-info" RuntimeError)
echo "Clearing virtualenv cache ..."
rm -rf "${HOME}/.local/share/virtualenv"

# Also remove any previous broken venv
rm -rf "${VENV_DIR}"

echo "Creating virtualenv at ${VENV_DIR} ..."
virtualenv --no-download "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"

# Install all dependencies from the CVMFS wheelhouse (--no-index = no network)
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
