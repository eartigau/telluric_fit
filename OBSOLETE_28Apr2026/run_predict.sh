#!/bin/bash
# Pull latest code and run predict_abso.py
# Only runs on FIR machine

cd "$(dirname "$0")"

# Check if we're on FIR
if [ ! -d "/home/eartigau/projects/rrg-rdoyon/eartigau/tapas_test_fit" ]; then
    echo "Error: This script should only be run on FIR"
    exit 1
fi

git pull
python predict_abso.py "$@"
