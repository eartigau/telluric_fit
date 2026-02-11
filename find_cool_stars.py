#!/usr/bin/env python3
"""
Find cool stars in scidata_NIRPS folder based on effective temperature.

This script:
1. Loops through all folders in scidata_NIRPS
2. Counts files in each folder
3. For folders with more than N files, reads the first science file header
4. Checks PP_TEFF keyword and prints folders with stars between 2500-4000K
"""

import os
import glob
import argparse
from astropy.io import fits


def find_cool_stars(min_files=5, teff_min=2500, teff_max=4000, data_dir='/project/6102120/eartigau/tapas/test_fit/scidata_NIRPS'):
    """
    Find folders containing cool stars based on effective temperature.
    
    Parameters
    ----------
    min_files : int
        Minimum number of files required in folder
    teff_min : float
        Minimum effective temperature (K)
    teff_max : float
        Maximum effective temperature (K)
    data_dir : str
        Path to data directory
    """
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found")
        return
    
    # Get all subdirectories
    folders = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    print(f"Scanning {len(folders)} folders in {data_dir}...")
    print(f"Criteria: >=  {min_files} files, {teff_min}K <= PP_TEFF <= {teff_max}K")
    print("-" * 80)
    
    results = []
    
    for folder in sorted(folders):
        folder_path = os.path.join(data_dir, folder)
        
        # Find all FITS files in the folder
        fits_files = glob.glob(os.path.join(folder_path, '*.fits'))
        n_files = len(fits_files)
        
        # Skip if not enough files
        if n_files < min_files:
            continue
        
        # Find first science file (typically contains 'pp_e2dsff' or similar)
        science_files = [f for f in fits_files if 'e2dsff' in f or 'e2ds' in f]
        
        if not science_files:
            # If no specific science files, use first FITS file
            science_files = fits_files
        
        if not science_files:
            continue
        
        # Read header of first science file
        try:
            with fits.open(science_files[0]) as hdul:
                header = hdul[0].header
                
                # Check for PP_TEFF keyword
                if 'PP_TEFF' in header:
                    teff = header['PP_TEFF']
                    
                    # Check if within range
                    if teff_min <= teff <= teff_max:
                        obj_name = header.get('OBJECT', 'UNKNOWN')
                        results.append({
                            'folder': folder,
                            'object': obj_name,
                            'teff': teff,
                            'n_files': n_files
                        })
                        print(f"{folder:25s}  Object: {obj_name:20s}  "
                              f"Teff: {teff:6.1f}K  Files: {n_files:4d}")
                else:
                    # PP_TEFF not found
                    pass
                    
        except Exception as e:
            print(f"Warning: Could not read {science_files[0]}: {e}")
            continue
    
    print("-" * 80)
    print(f"\nFound {len(results)} folders matching criteria")
    
    if results:
        print("\n# Copy-paste into batch_config.yaml objects list:")
        print("objects:")
        for res in results:
            print(f"  - name: '{res['folder']}'")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Find cool stars in NIRPS data based on effective temperature'
    )
    parser.add_argument(
        '-n', '--min-files',
        type=int,
        default=5,
        help='Minimum number of files in folder (default: 5)'
    )
    parser.add_argument(
        '--teff-min',
        type=float,
        default=2500,
        help='Minimum effective temperature in K (default: 2500)'
    )
    parser.add_argument(
        '--teff-max',
        type=float,
        default=4000,
        help='Maximum effective temperature in K (default: 4000)'
    )
    parser.add_argument(
        '-d', '--data-dir',
        type=str,
        default='/project/6102120/eartigau/tapas/test_fit/scidata_NIRPS',
        help='Data directory path (default: /project/6102120/eartigau/tapas/test_fit/scidata_NIRPS)'
    )
    
    args = parser.parse_args()
    
    find_cool_stars(
        min_files=args.min_files,
        teff_min=args.teff_min,
        teff_max=args.teff_max,
        data_dir=args.data_dir
    )


if __name__ == '__main__':
    main()
