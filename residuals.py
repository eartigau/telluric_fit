import glob
import os
import warnings
from astropy.io import fits
import numpy as np
from astropy.table import Table
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tellu_tools import hotstar, construct_abso, getdata_safe, savgol_filter_nan_fast, savgol_filter_robust, getheader_safe, load_telluric_config
from tellu_tools_config import get_user_params
from aperocore import math as mp
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from aperocore.science import wavecore
from scipy.signal import medfilt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
import time
import socket

# Number of parallel workers - use 6 on MacBook (memory constrained), 8 elsewhere
_hostname = socket.gethostname()
if 'MacBook' in _hostname or 'macbook' in _hostname.lower():
    N_WORKERS = 6
else:
    N_WORKERS = min(8, multiprocessing.cpu_count())


def _fast_berv_shift(residuals, wave, bervs, to_stellar=True):
    """
    Fast vectorized BERV shift for multiple exposures.
    
    Parameters
    ----------
    residuals : ndarray, shape (n_exp, n_pix)
        Residual spectra to shift
    wave : ndarray, shape (n_pix,)
        Wavelength grid
    bervs : array-like, shape (n_exp,)
        BERV values for each exposure (km/s)
    to_stellar : bool
        If True, shift to stellar frame (apply -BERV)
        If False, shift from stellar frame (apply +BERV)
    
    Returns
    -------
    shifted : ndarray, shape (n_exp, n_pix)
        Shifted residuals
    """
    n_exp, n_pix = residuals.shape
    shifted = np.zeros_like(residuals)
    
    # Pre-compute relativistic factors for all BERVs
    sign = -1.0 if to_stellar else 1.0
    
    for i in range(n_exp):
        berv = bervs[i]
        # Target wavelength grid
        wave_target = wave * mp.relativistic_waveshift(sign * berv)
        # Use linear interpolation (much faster than spline for this application)
        valid = np.isfinite(residuals[i])
        if np.sum(valid) < 10:
            shifted[i] = np.nan
            continue
        # Interpolate from original grid to target grid
        shifted[i] = np.interp(wave, wave_target, residuals[i], left=np.nan, right=np.nan)
    
    return shifted


def _load_header_only(file_and_keys):
    """Load header keywords from a single FITS file (no pixel data). Used for parallel loading."""
    file, keys = file_and_keys
    hdr = getheader_safe(file)
    hdr = hotstar(hdr)
    header_vals = {key: hdr[key] for key in keys}
    return header_vals


def _load_order_from_file(file_and_iord):
    """Load a single spectral order (one row) from a FITS file."""
    file, iord = file_and_iord
    return getdata_safe(file)[iord]


def _process_single_order(args):
    """
    Process a single spectral order for residual analysis.
    
    This function is designed to be called in parallel via ProcessPoolExecutor.
    
    Parameters
    ----------
    args : tuple
        (iord, order_data, wave_order, main_abso_order, nanmask_order, tbl0_dict, residuals_dir)
    
    Returns
    -------
    dict with keys: iord, slope_offset, dc_offset, rms, rms_envelope
    """
    iord, order_data, wave_order, main_abso_order, nanmask_order, tbl0_dict, residuals_dir = args
    
    # Reconstruct table from dict (can't pickle astropy Table directly in some cases)
    tbl = Table(tbl0_dict)
    
    outname1 = os.path.join(residuals_dir, f'residuals_order_{iord:02d}_slope.fits')
    outname2 = os.path.join(residuals_dir, f'residuals_order_{iord:02d}_intercept.fits')
    outname3 = os.path.join(residuals_dir, f'residuals_order_{iord:02d}_rms.fits')
    outname4 = os.path.join(residuals_dir, f'residuals_order_{iord:02d}_rms_envelope.fits')
    
    # Pre-allocate arrays for residuals and wavelength vectors per exposure
    residuals = order_data.T * nanmask_order  # shape (nexp, npix)
    
    # Remove global DC offset across exposures to center residuals
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        residuals -= np.nanmedian(residuals)
    
    # -------------------------------------------------------------------------
    # Per-object alignment and removal of the common residual pattern
    # -------------------------------------------------------------------------
    for uobj in np.unique(tbl['DRSOBJN']):
        g = tbl['DRSOBJN'] == uobj
        g_indices = np.where(g)[0]
        bervs_obj = np.array(tbl['BERV'][g])
        
        # Shift residuals TO stellar rest frame using fast vectorized function
        residual_tmp = _fast_berv_shift(residuals[g], wave_order, bervs_obj, to_stellar=True)
        
        # Build a median residual pattern in the stellar rest frame
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(residual_tmp, axis=0)
            
            # Iteratively detrend median residuals to remove any slope
            for i in range(residual_tmp.shape[0]):
                diff = residual_tmp[i] - med
                try:
                    fit = mp.robust_polyfit(wave_order, diff, 1, 3)[0]
                    off = np.polyval(fit, wave_order)
                    residual_tmp[i] -= off
                except Exception:
                    # Skip detrending if not enough valid points
                    pass
            med = np.nanmedian(residual_tmp, axis=0)
        
        # Median filter to remove outliers/single-pixel spikes
        med = medfilt(med, kernel_size=11)
        # Smooth it to remove high-frequency noise (handles NaNs)
        med_filt = savgol_filter_nan_fast(med, 101, 3, frac_valid=0.3)
        
        # Compute corrections for all exposures at once
        med_filt_repeated = np.tile(med_filt, (len(bervs_obj), 1))
        corrections = _fast_berv_shift(med_filt_repeated, wave_order, bervs_obj, to_stellar=False)
        
        # Apply corrections and detrending
        for i in range(len(g_indices)):
            idx = g_indices[i]
            residuals[idx] -= corrections[i]
            
            # Detrend each exposure with robust savgol filter (~100 km/s scale)
            trend = savgol_filter_robust(residuals[idx], window_length=101, polyorder=3, n_sigma=5.0)
            residuals[idx] -= np.nan_to_num(trend, nan=0.0)
    
    # -------------------------------------------------------------------------
    # Exposure filtering
    # -------------------------------------------------------------------------
    tbl['DOY'] = tbl['MJDMID'] % 365.24
    
    # Keep only hot stars and moderate water content to stabilize fits
    keep = tbl['HOTSTAR'] & (tbl['EXPO_H2O'] < 7.0)
    tbl = tbl[keep]
    residuals = residuals[keep, :]
    
    # -------------------------------------------------------------------------
    # Per-pixel linear modeling of residuals vs drivers (H2O or AIRMASS)
    # -------------------------------------------------------------------------
    dc_offset = np.full(residuals.shape[1], np.nan)
    slope_offset = np.full(residuals.shape[1], np.nan)
    recon = np.full(residuals.shape, np.nan)
    
    # Identify valid pixels (>50% finite values)
    valid_frac = np.mean(np.isfinite(residuals), axis=0)
    valid_pix = valid_frac >= 0.5
    
    # Separate pixels by absorber type
    h2o_mask = (main_abso_order == 0) | (main_abso_order == 4)
    o2_mask = np.isin(main_abso_order, [1, 2, 3])
    morning = np.array(tbl['SUNSETD'] < 5.0)
    
    # --- Robust per-pixel linear fit for H2O/None pixels ---
    h2o_pix = valid_pix & h2o_mask
    if np.any(h2o_pix):
        x = np.array(tbl['EXPO_H2O'])
        h2o_indices = np.where(h2o_pix)[0]
        for ipix in h2o_indices:
            y = residuals[:, ipix]
            valid = np.isfinite(y)
            if np.sum(valid) < 3:
                continue
            try:
                fit, _ = mp.robust_polyfit(x[valid], y[valid], 1, 3)
                slope_offset[ipix] = fit[0]
                dc_offset[ipix] = fit[1]
                recon[:, ipix] = fit[1] + fit[0] * x
            except Exception:
                pass
    
    # --- Robust per-pixel linear fit for O2/CO2/CH4 pixels (morning only) ---
    o2_pix = valid_pix & o2_mask
    if np.any(o2_pix):
        x_full = np.array(tbl['AIRMASS'])
        x = x_full[morning]
        o2_indices = np.where(o2_pix)[0]
        for ipix in o2_indices:
            y = residuals[morning, ipix]
            valid = np.isfinite(y)
            if np.sum(valid) < 3:
                continue
            try:
                fit, _ = mp.robust_polyfit(x[valid], y[valid], 1, 3)
                slope_offset[ipix] = fit[0]
                dc_offset[ipix] = fit[1]
                recon[:, ipix] = fit[1] + fit[0] * x_full
            except Exception:
                pass
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        n1, p1 = np.nanpercentile(residuals - recon, [16, 84], axis=0)
    rms = (p1 - n1) / 2.0
    
    # Compute smoothed RMS envelope for adaptive thresholding
    rms_med = medfilt(rms, kernel_size=11)
    rms_envelope = savgol_filter_nan_fast(rms_med, 101, 3, frac_valid=0.3)
    
    # Write output files
    fits.writeto(outname1, slope_offset, overwrite=True)
    fits.writeto(outname2, dc_offset, overwrite=True)
    fits.writeto(outname3, rms, overwrite=True)
    fits.writeto(outname4, rms_envelope, overwrite=True)
    
    return {
        'iord': iord,
        'slope_offset': slope_offset,
        'dc_offset': dc_offset,
        'rms': rms,
        'rms_envelope': rms_envelope,
        'residuals': residuals,   # processed, BERV-aligned, detrended; shape (n_exp_filtered, n_pix)
        'tbl': tbl,               # filtered table (HOTSTAR & EXPO_H2O < 7)
    }

# -----------------------------------------------------------------------------
# This script builds a per-pixel, per-order residual model for telluric
# transmission fits. It:
#   1) Loads per-exposure fitted transmission products (trans_*.fits)
#   2) Aligns residuals to a common wavelength grid (per object) using BERV
#   3) Removes a common median residual per object (with optional example plots)
#   4) Detrends each exposure by a robust linear fit vs wavelength
#   5) For each pixel, fits residuals as linear functions of:
#         - EXPO_H2O for H2O/no-absorption pixels
#         - AIRMASS for O2 pixels (morning-only)
#   6) Saves per-order maps of slope and intercept as FITS
#
# All plotting is controlled via the global "doplot" flag.
# -----------------------------------------------------------------------------

# Coding errors (deprecated APIs, future incompatibilities) should crash immediately
# rather than silently producing NaN-filled outputs.
warnings.filterwarnings('error', category=DeprecationWarning)
warnings.filterwarnings('error', category=FutureWarning)
# Suppress only the specific expected NaN-related runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*All-NaN.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')

doplot = False  # Set to True to enable all diagnostic plots

# Paper figure tracking
_paper_figure_done = {'fig3': False, 'fig4': False, 'fig5': False}


def get_paper_figures_config():
    """Get paper figures configuration from yaml."""
    config = load_telluric_config()
    paper_config = config.get('paper_figures', {})
    enabled = paper_config.get('enabled', False)
    
    if not enabled:
        return False, None
    
    output_dir = os.path.join(project_path, paper_config.get('output_dir', 'paper_figures'))
    os.makedirs(output_dir, exist_ok=True)
    return True, output_dir


def _generate_paper_fig3_berv_alignment(wave, residual_tmp, med, med_filt, obj_name, output_dir):
    """Generate paper figure 3: BERV-aligned stellar template removal.
    
    Shows individual residuals shifted to stellar frame, the median pattern,
    and the smoothed template that gets subtracted.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Panel 1: Individual residuals in stellar frame (waterfall-style)
    n_spectra = min(residual_tmp.shape[0], 20)  # Limit for clarity
    for i in range(n_spectra):
        offset = i * 0.02  # Small vertical offset for visibility
        axes[0].plot(wave, residual_tmp[i] + offset, 'k-', lw=0.3, alpha=0.5)
    
    axes[0].set_ylabel('Residuals (offset)')
    axes[0].set_title(f'BERV-Aligned Residuals in Stellar Frame - {obj_name}')
    
    # Panel 2: Median and smoothed template
    axes[1].plot(wave, med, 'b-', lw=0.5, alpha=0.7, label='Median')
    axes[1].plot(wave, med_filt, 'r-', lw=1.5, label='Smoothed (Savgol)')
    axes[1].axhline(0, color='k', ls='--', lw=0.5, alpha=0.5)
    axes[1].set_ylabel('Median Residual')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'fig3_berv_alignment.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Paper figure saved: {fig_path}')


def _generate_paper_fig4_residual_model(wave, residuals, tbl, slope_offset, dc_offset, 
                                        main_abso_order, output_dir):
    """Generate paper figure 4: Per-pixel residual model fits.
    
    Shows scatter of residuals vs EXPO_H2O for example pixels,
    with fitted linear trends.
    """
    # Select 4 example pixels at different wavelengths, restricted to H2O/None
    # absorber pixels (main_abso == 0 or 4) — these are the ones whose slope was
    # fit against EXPO_H2O. O2/CO2/CH4 pixels are fit against AIRMASS, so plotting
    # them against EXPO_H2O would show a misleadingly flat line.
    npix = len(wave)
    h2o_or_none = (main_abso_order == 0) | (main_abso_order == 4)
    valid_pix_for_plot = np.where(h2o_or_none & np.isfinite(slope_offset))[0]

    if len(valid_pix_for_plot) < 4:
        print(f'  fig4: only {len(valid_pix_for_plot)} valid H2O/None pixels — skipping figure')
        return

    # Pick 4 pixels evenly spaced along the order in wavelength
    quantiles = [0.2, 0.4, 0.6, 0.8]
    pix_indices = [valid_pix_for_plot[int(q * (len(valid_pix_for_plot) - 1))]
                   for q in quantiles]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    expo_h2o = np.array(tbl['EXPO_H2O'])
    
    for ax, ipix in zip(axes, pix_indices):
        y = residuals[:, ipix]
        valid = np.isfinite(y)
        
        if np.sum(valid) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
            continue
        
        ax.scatter(expo_h2o[valid], y[valid], s=10, alpha=0.5, c='blue')
        
        # Plot fitted line
        slope = slope_offset[ipix]
        intercept = dc_offset[ipix]
        if np.isfinite(slope) and np.isfinite(intercept):
            print(f'Pixel {ipix} ({wave[ipix]:.1f} nm): slope={slope:.4f}, intercept={intercept:.4f}')
            x_fit = np.linspace(np.min(expo_h2o), np.max(expo_h2o), 100)
            y_fit = intercept + slope * x_fit
            ax.plot(x_fit, y_fit, 'r-', lw=2, label=f'slope={slope:.4f}')
            ax.legend(loc='upper right', fontsize=8)
        
        ax.axhline(0, color='k', ls='--', lw=0.5, alpha=0.5)
        ax.set_xlabel('EXPO_H2O')
        ax.set_ylabel('Residual')
        ax.set_title(f'Pixel {ipix} ({wave[ipix]:.1f} nm)')
    
    plt.suptitle('Residual vs H2O Exponent - Per-Pixel Linear Fits')
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'fig4_residual_model.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Paper figure saved: {fig_path}')


if __name__ == '__main__':
    # ==========================================================================
    # MAIN EXECUTION BLOCK
    # ==========================================================================
    # This block must be guarded to prevent re-execution when ProcessPoolExecutor
    # spawns child processes (required on macOS which uses 'spawn' method)
    
    # -------------------------------------------------------------------------
    # Load inputs and prepare global/static products
    # -------------------------------------------------------------------------

    instrument = 'NIRPS'

    # Get project path for this machine
    params = get_user_params(instrument)
    project_path = params['project_path']

    # Main absorber map (e.g. 0: H2O, 1: O2, etc.) over the common reference wave grid
    main_abso = getdata_safe(os.path.join(project_path, f'main_absorber_{instrument}.fits'))

    # Reference/common wave grid used for alignment and plotting diagnostics
    waveref = getdata_safe(os.path.join(project_path, f'calib_{instrument}/waveref.fits'))

    # Build the baseline absorption cube for TAPAS (H2O, CO2, CH4, O2) on wave0
    # Using expos=[1,1,1,1] returns per-molecule normalized absorption arrays
    all_abso = construct_abso(waveref, [1,1,1,1], all_abso=None)

    # Mean absorber across molecules (product along the first axis)
    # Used for masking out near-transparent regions (low absorption)
    mean_abso = np.prod(all_abso, axis=0)

    # Sanity check: mean_abso must have finite values, otherwise all output maps
    # will be filled with NaN/zeros silently (e.g. if construct_abso returned all-NaN).
    if not np.any(np.isfinite(mean_abso)):
        raise ValueError('mean_abso is entirely NaN/Inf — construct_abso returned bad data. '
                         'Check that the TAPAS absorption cube is loaded correctly.')

    # Mask out pixels with little/no absorption to reduce noise amplification
    nanmask = np.ones(mean_abso.shape, dtype=float)
    nanmask[mean_abso < 0.3] = np.nan

    # Sanity check: at least some pixels must be unmasked (absorption > 0.3)
    if not np.any(np.isfinite(nanmask)):
        raise ValueError('nanmask is entirely NaN — no pixels have mean absorption > 0.3. '
                         'Check the absorption threshold and the TAPAS data.')


    # Output maps (per order, per pixel): slope and intercept for residual trends
    map_slopes = np.zeros_like(mean_abso)
    map_intercepts = np.zeros_like(mean_abso)
    map_rms = np.zeros_like(mean_abso)
    map_rms_envelope = np.zeros_like(mean_abso)  # Smoothed RMS envelope for threshold

    # List of all fitted transmissions to analyze
    files = glob.glob(os.path.join(project_path, f'tellu_fit_{instrument}/trans_*.fits'))  # consider subsampling with [::N] if needed

    # Build a small table with metadata we need for conditioning and coloring
    tbl0 = Table()
    tbl0['FILE'] = files

    keys = ['AIRMASS', 'DRSOBJN', 'HOTSTAR', 'EXPO_H2O', 'EXPO_O2',
            'TEMPERAT', 'PRESSURE', 'HUMIDITY', 'MJDMID', 'SUNSETD', 'H2O_CV','BERV']

    for key in keys:
        # strings initially, will attempt cast to float/bool later
        tbl0[key] = np.zeros(len(files), dtype='U999')


    # -------------------------------------------------------------------------
    # Pass 1: load headers only (no pixel data) — memory-efficient
    # -------------------------------------------------------------------------
    print(f'Loading headers from {len(files)} FITS files using {N_WORKERS} workers...')
    file_args = [(f, keys) for f in files]
    header_results = [None] * len(files)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        future_to_idx = {executor.submit(_load_header_only, args): i for i, args in enumerate(file_args)}
        for j, future in enumerate(tqdm(as_completed(future_to_idx), total=len(files),
                                        desc='Loading headers', unit='files',
                                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')):
            idx = future_to_idx[future]
            header_results[idx] = future.result()

    elapsed = time.time() - t0
    print(f'  Finished loading headers in {elapsed:.1f}s ({len(files)/elapsed:.1f} files/s)')

    for i, header_vals in enumerate(header_results):
        for key in keys:
            tbl0[key][i] = header_vals[key]

    # Attempt to cast columns to numeric or boolean where applicable
    for key in keys:
        try:
            tbl0[key] = tbl0[key].astype(float)
        except Exception:
            tbl0[key] = tbl0[key].astype(str)
        # Convert string booleans to True/False if column contains only them
        if np.all(np.isin(tbl0[key], ['True', 'False'])):
            tbl0[key] = tbl0[key] == 'True'

    # bad is where EXPO_H2O <0.01 (shouldn't happen but just in case)
    bad = tbl0['EXPO_H2O'] < 0.01
    if np.any(bad):
        print(f'  WARNING: Found {np.sum(bad)} exposures with EXPO_H2O < 0.01. These will be excluded from analysis.')
        tbl0 = tbl0[~bad]   
        # Update files list to match filtered table
        files = tbl0['FILE'].tolist()


    # -------------------------------------------------------------------------
    # Pass 2+3: for each order, load only that order's data then process it.
    # Peak memory is O(n_files * n_pix) instead of O(n_orders * n_pix * n_files).
    # -------------------------------------------------------------------------
    residuals_dir = os.path.join(project_path, f'residuals_{instrument}')
    os.makedirs(residuals_dir, exist_ok=True)

    tbl0_dict = {col: np.array(tbl0[col]) for col in tbl0.colnames}

    n_orders = waveref.shape[0]
    print(f'Processing {n_orders} orders (loading one order at a time to save memory)...')

    order_results = []
    t0_orders = time.time()
    for iord in tqdm(range(n_orders), desc='Processing orders', unit='order',
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
        # Load this order from all files in parallel (shape: n_files x n_pix)
        order_data = np.zeros((len(files), waveref.shape[1]))
        order_file_args = [(f, iord) for f in files]
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            future_to_idx = {executor.submit(_load_order_from_file, args): i
                             for i, args in enumerate(order_file_args)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                order_data[idx] = future.result()

        # _process_single_order expects order_data shaped (n_pix, n_files)
        args = (
            iord,
            order_data.T,
            waveref[iord, :],
            main_abso[iord, :],
            nanmask[iord, :],
            tbl0_dict,
            residuals_dir
        )
        result = _process_single_order(args)
        order_results.append(result)

        map_slopes[iord, :] = result['slope_offset']
        map_intercepts[iord, :] = result['dc_offset']
        map_rms[iord, :] = result['rms']
        map_rms_envelope[iord, :] = result['rms_envelope']

        # Generate paper figure 4 after order 0 using the fully processed residuals
        enabled, output_dir_pf = get_paper_figures_config()
        if enabled and not _paper_figure_done['fig4'] and iord == 0:
            _generate_paper_fig4_residual_model(
                waveref[iord, :], result['residuals'], result['tbl'],
                map_slopes[iord, :], map_intercepts[iord, :],
                main_abso[iord, :], output_dir_pf
            )
            _paper_figure_done['fig4'] = True

    elapsed_orders = time.time() - t0_orders
    print(f'  Finished processing orders in {elapsed_orders:.1f}s ({n_orders/elapsed_orders:.2f} orders/s)')

    # -------------------------------------------------------------------------
    # Generate multi-page PDF summary
    # -------------------------------------------------------------------------
    print('Generating multi-page PDF summary...')

    # Molecule names and colors
    molecule_names = {0: 'H2O', 1: 'O2', 2: 'CO2', 3: 'CH4', 4: 'None'}
    molecule_colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange', 4: 'grey'}

    pdf_path = os.path.join(residuals_dir, f'residuals_summary_{instrument}.pdf')
    with PdfPages(pdf_path) as pdf:
        # Page 1: Overview of all orders - slope map
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Flatten wavelength for x-axis
        wave_flat = waveref.flatten()
        order_idx = np.argsort(wave_flat)
        wave_sorted = wave_flat[order_idx]
        
        slope_flat = map_slopes.flatten()[order_idx]
        intercept_flat = map_intercepts.flatten()[order_idx]
        rms_flat = map_rms.flatten()[order_idx]
        main_abso_flat = main_abso.flatten()[order_idx]
        
        # Color by main absorber (rasterized for faster PDF rendering)
        for mol_id in range(5):
            mask = main_abso_flat == mol_id
            if np.any(mask):
                axes[0].scatter(wave_sorted[mask], slope_flat[mask], s=1, alpha=0.5,
                              c=molecule_colors[mol_id], label=molecule_names[mol_id],
                              rasterized=True)
                axes[1].scatter(wave_sorted[mask], intercept_flat[mask], s=1, alpha=0.5,
                              c=molecule_colors[mol_id], rasterized=True)
        
        axes[0].set_ylabel('Slope')
        axes[0].set_title(f'Residual Slope by Main Absorber - {instrument}')
        axes[0].legend(loc='upper right', markerscale=5)
        axes[0].set_ylim(np.nanpercentile(slope_flat, [1, 99]))
        
        axes[1].set_ylabel('Intercept')
        axes[1].set_title('Residual Intercept by Main Absorber')
        axes[1].set_ylim(np.nanpercentile(intercept_flat, [1, 99]))
        
        # Mean absorption for context
        mean_abso_flat = mean_abso.flatten()[order_idx]
        axes[2].plot(wave_sorted, mean_abso_flat, 'k-', lw=0.3, alpha=0.5, rasterized=True)
        axes[2].set_ylabel('Mean Absorption')
        axes[2].set_xlabel('Wavelength (nm)')
        axes[2].set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        # Paper Figure 5: Full slope/intercept overview (only generated once)
        enabled, output_dir = get_paper_figures_config()
        if enabled and not _paper_figure_done['fig5']:
            fig_path = os.path.join(output_dir, 'fig5_residual_slopes_overview.pdf')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f'Paper figure saved: {fig_path}')
            _paper_figure_done['fig5'] = True
        
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        
        # One page per molecule showing detailed view
        for mol_id, mol_name in molecule_names.items():
            mask = main_abso_flat == mol_id
            if not np.any(mask):
                continue
                
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            axes[0].scatter(wave_sorted[mask], slope_flat[mask], s=2, alpha=0.5,
                           c=molecule_colors[mol_id], rasterized=True)
            axes[0].axhline(0, color='k', ls='--', lw=0.5)
            axes[0].set_ylabel('Slope')
            axes[0].set_title(f'{mol_name} Pixels - Residual Slope ({np.sum(mask)} pixels)')
            axes[0].set_ylim(np.nanpercentile(slope_flat[mask], [1, 99]))
            
            axes[1].scatter(wave_sorted[mask], intercept_flat[mask], s=2, alpha=0.5,
                           c=molecule_colors[mol_id], rasterized=True)
            axes[1].axhline(0, color='k', ls='--', lw=0.5)
            axes[1].set_ylabel('Intercept')
            axes[1].set_title(f'{mol_name} Pixels - Residual Intercept')
            axes[1].set_ylim(np.nanpercentile(intercept_flat[mask], [1, 99]))
            
            # Histogram of slopes
            axes[2].hist(slope_flat[mask][np.isfinite(slope_flat[mask])], bins=50, 
                        color=molecule_colors[mol_id], alpha=0.7)
            axes[2].set_xlabel('Slope')
            axes[2].set_ylabel('Count')
            axes[2].set_title(f'{mol_name} Slope Distribution')
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
        
        # One page per order showing slope, intercept, and RMS
        print('  Adding per-order pages...')
        
        # Load RMS excess factor from config
        telluric_config = load_telluric_config()
        rms_excess_factor = telluric_config.get('quality_control', {}).get('rms_excess_factor', 2.0)
        
        # Validate the excess factor (must be >= 1.5)
        if rms_excess_factor < 1.5:
            raise ValueError(f'rms_excess_factor ({rms_excess_factor}) must be >= 1.5. '
                            f'Lower values would reject too many valid pixels.')
        
        for iord in tqdm(range(waveref.shape[0]), desc='Adding order pages'):
            wave_ord = waveref[iord, :]
            slope_ord = map_slopes[iord, :]
            intercept_ord = map_intercepts[iord, :]
            rms_ord = map_rms[iord, :]
            rms_envelope_ord = map_rms_envelope[iord, :]
            main_abso_ord = main_abso[iord, :]
            
            # Compute adaptive threshold: envelope * factor
            rms_threshold = rms_envelope_ord * rms_excess_factor
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Add pink shading for excess RMS regions (above envelope * factor)
            excess_rms = rms_ord > rms_threshold
            for ax in axes:
                ax.fill_between(wave_ord, 0, 1, where=excess_rms,
                               color='pink', alpha=0.3, transform=ax.get_xaxis_transform(),
                               rasterized=True)
            
            # Plot slope colored by main absorber
            for mol_id in range(5):
                mask = main_abso_ord == mol_id
                if np.any(mask):
                    axes[0].scatter(wave_ord[mask], slope_ord[mask], s=4, alpha=0.7,
                                  c=molecule_colors[mol_id], label=molecule_names[mol_id],
                                  rasterized=True)
            axes[0].axhline(0, color='k', ls='--', lw=0.5)
            axes[0].set_ylabel('Slope')
            axes[0].set_title(f'Order {iord} - Residual Slope vs Wavelength')
            # Add legend with shading entries
            from matplotlib.patches import Patch
            handles, labels = axes[0].get_legend_handles_labels()
            handles.append(Patch(facecolor='pink', alpha=0.3, label=f'Excess RMS (>{rms_excess_factor}x envelope)'))
            axes[0].legend(handles=handles, loc='upper right', markerscale=3)
            ylim = np.nanpercentile(slope_ord, [1, 99])
            if np.isfinite(ylim).all():
                axes[0].set_ylim(ylim)
            
            # Plot intercept colored by main absorber
            for mol_id in range(5):
                mask = main_abso_ord == mol_id
                if np.any(mask):
                    axes[1].scatter(wave_ord[mask], intercept_ord[mask], s=4, alpha=0.7,
                                  c=molecule_colors[mol_id], rasterized=True)
            axes[1].axhline(0, color='k', ls='--', lw=0.5)
            axes[1].set_ylabel('Intercept')
            axes[1].set_title(f'Order {iord} - Residual Intercept vs Wavelength')
            ylim = np.nanpercentile(intercept_ord, [1, 99])
            if np.isfinite(ylim).all():
                axes[1].set_ylim(ylim)
            
            # Plot RMS colored by main absorber
            for mol_id in range(5):
                mask = main_abso_ord == mol_id
                if np.any(mask):
                    axes[2].scatter(wave_ord[mask], rms_ord[mask], s=4, alpha=0.7,
                                  c=molecule_colors[mol_id], rasterized=True)
            # Plot the smoothed envelope and threshold
            axes[2].plot(wave_ord, rms_envelope_ord, 'k-', lw=1.5, alpha=0.7, label='Envelope')
            axes[2].plot(wave_ord, rms_threshold, 'r--', lw=1.5, alpha=0.7, 
                        label=f'Threshold ({rms_excess_factor}x)')
            axes[2].set_ylabel('RMS')
            axes[2].set_xlabel('Wavelength (nm)')
            axes[2].set_title(f'Order {iord} - Residual RMS vs Wavelength')
            axes[2].legend(loc='upper right', fontsize=8)
            rms_median = np.nanmedian(rms_ord)
            if np.isfinite(rms_median):
                axes[2].set_ylim(0, 8 * rms_median)
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f'PDF saved to {pdf_path}')
