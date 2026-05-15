#!/usr/bin/env bash
# Install third-party dependencies for telluric_fit
# Designed for Compute Canada (Narval/FIR) with CVMFS wheelhouse.
#
# Usage:
#   module load python/3.12.4
#   ./install_deps.sh
#   source ~/telluric_env/bin/activate

set -e

VENV_DIR="${HOME}/telluric_env"
WHEELHOUSE="/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic"

rm -rf "${VENV_DIR}"

echo "Creating bare virtualenv (no pip seeding) ..."
# --without-pip avoids the filelock/I-O error in virtualenv's seed mechanism
python -m venv --without-pip "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"

# Bootstrap pip by running it directly from the CVMFS wheel (a zip file)
PIP_WHEEL=$(ls "${WHEELHOUSE}"/pip-*.whl 2>/dev/null | sort -V | tail -1)
if [ -z "${PIP_WHEEL}" ]; then
    echo "ERROR: no pip wheel found in ${WHEELHOUSE}" >&2
    exit 1
fi
echo "Bootstrapping pip from ${PIP_WHEEL} ..."
python "${PIP_WHEEL}/pip" install --no-index "${PIP_WHEEL}"

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
