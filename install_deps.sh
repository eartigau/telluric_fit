#!/usr/bin/env bash
# Install third-party dependencies for telluric_fit (compil_stats, tellu_tools, predict_abso, etc.)

pip install \
    numpy \
    matplotlib \
    astropy \
    scipy \
    tqdm \
    numexpr \
    PyYAML
