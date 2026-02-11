#!/bin/bash

SRC_BASE=~/scratch/nirps_tempo/objects
DEST_SCI=/home/eartigau/projects/rrg-rdoyon/eartigau/tapas/test_fit/scidata_NIRPS
DEST_ORIG=/home/eartigau/projects/rrg-rdoyon/eartigau/tapas/test_fit/orig_NIRPS

for dir in "$SRC_BASE"/*/; do
    OBJECT=$(basename "$dir")
    
    # Compter les fichiers t.fits
    count=$(find "$dir" -maxdepth 1 -name "*t.fits" | wc -l)
    
    if [ "$count" -gt 50 ]; then
        echo "=== $OBJECT ($count fichiers t.fits) ==="
        
        mkdir -p "${DEST_SCI}/${OBJECT}"
        mkdir -p "${DEST_ORIG}/${OBJECT}"
        
        rsync -artvh "${dir}"*_pp_e2dsff_A.fits "${DEST_SCI}/${OBJECT}/"
        rsync -artvh "${dir}"*t.fits "${DEST_ORIG}/${OBJECT}/"
    fi
done