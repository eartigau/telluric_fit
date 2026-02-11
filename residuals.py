import glob
import os
import warnings
from astropy.io import fits
import numpy as np
from astropy.table import Table
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tellu_tools import hotstar, construct_abso, getdata_safe, savgol_filter_nan_fast, getheader_safe, load_telluric_config
from tellu_tools_config import get_user_params
from aperocore import math as mp
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from aperocore.science import wavecore
from scipy.signal import medfilt

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

# Suppress RuntimeWarnings for NaN operations (expected with masked data)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*All-NaN.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')

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
    # Select 4 example pixels at different wavelengths
    npix = len(wave)
    pix_indices = [npix//5, 2*npix//5, 3*npix//5, 4*npix//5]
    
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


# -----------------------------------------------------------------------------
# Load inputs and prepare global/static products
# -----------------------------------------------------------------------------


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
mean_abso = np.product(all_abso, axis=0)

# Mask out pixels with little/no absorption to reduce noise amplification
nanmask = np.ones(mean_abso.shape, dtype=float)
nanmask[mean_abso < 0.3] = np.nan


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


big_cube = np.zeros((waveref.shape[0], waveref.shape[1], len(files)))
for i, file in enumerate(tqdm(files, desc='Building big cube', leave=False, unit='files')):
    trans = getdata_safe(file)

    big_cube[:, :, i] = trans


    hdr = getheader_safe(file)

    # Classify hot star for filtering later
    hdr = hotstar(hdr)

    # Mirror selected header keywords into the table
    for key in keys:
        tbl0[key][i] = hdr[key]

# Attempt to cast columns to numeric or boolean where applicable
for key in keys:
    try:
        tbl0[key] = tbl0[key].astype(float)
    except Exception:
        tbl0[key] = tbl0[key].astype(str)
    # Convert string booleans to True/False if column contains only them
    if np.all(np.isin(tbl0[key], ['True', 'False'])):
        tbl0[key] = tbl0[key] == 'True'


# -----------------------------------------------------------------------------
# Process residuals order-by-order
# -----------------------------------------------------------------------------
residuals_dir = os.path.join(project_path, f'residuals_{instrument}')
os.makedirs(residuals_dir, exist_ok=True)

for iord in tqdm(range(waveref.shape[0]), desc='Processing orders', unit='order'):
    
    outname1 = os.path.join(residuals_dir, f'residuals_order_{iord:02d}_slope.fits')
    outname2 = os.path.join(residuals_dir, f'residuals_order_{iord:02d}_intercept.fits')
    outname3 = os.path.join(residuals_dir, f'residuals_order_{iord:02d}_rms.fits')
    outname4 = os.path.join(residuals_dir, f'residuals_order_{iord:02d}_rms_envelope.fits')

    tbl = Table(tbl0)

    # Extract per-order slices for speed
    main_abso_order = main_abso[iord, :]
    mean_abso_order = mean_abso[iord, :]

    # Pre-allocate arrays for residuals and wavelength vectors per exposure
    residuals = big_cube[iord, :, :].T * nanmask[iord, :]  # shape (nexp, npix)

    # Remove global DC offset across exposures to center residuals
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        residuals -= np.nanmedian(residuals)

    # -------------------------------------------------------------------------
    # Per-object alignment and removal of the common residual pattern
    # (to mitigate object-specific spectral features leaking into residuals)
    # -------------------------------------------------------------------------
    for uobj in np.unique(tbl['DRSOBJN']):
        g = tbl['DRSOBJN'] == uobj

        # Shift residuals TO stellar rest frame (apply -BERV to blueshift) before taking median
        residual_tmp = np.zeros(residuals[g].shape)
        for i in range(residuals[g].shape[0]):
            berv = tbl['BERV'][g][i]
            # Observation wavelength grid
            wave_obs = np.array([waveref[iord, :]])
            # Stellar rest frame wavelength grid (shift by -BERV to undo Doppler)
            wave_stellar = np.array([waveref[iord, :] * mp.relativistic_waveshift(-berv)])
            # Resample residuals from observation frame to stellar rest frame
            sp_obs = np.array([residuals[g][i]])
            residual_tmp[i] = wavecore.wave_to_wave(sp_obs, wave_obs, wave_stellar)[0]

        # Build a median residual pattern in the stellar rest frame
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(residual_tmp, axis=0)
        med = medfilt(med, kernel_size=11)
        # Smooth it to remove high-frequency noise (handles NaNs)
        med_filt = savgol_filter_nan_fast(med, 101, 3, frac_valid=0.3)

        # Paper Figure 3: BERV-aligned stellar template removal (only on first object, first order)
        enabled, output_dir = get_paper_figures_config()
        if enabled and not _paper_figure_done['fig3'] and iord == 0:
            _generate_paper_fig3_berv_alignment(
                waveref[iord, :], residual_tmp, med, med_filt, uobj, output_dir
            )
            _paper_figure_done['fig3'] = True

        # Optional: visualize a single sample correction application
        sample = False
        for i in range(residuals[g].shape[0]):
            # Pick one first exposure of the first object for demo plotting
            if (uobj == tbl['DRSOBJN'][0]) and (i == 0):
                sample = True

            berv = tbl['BERV'][g][i]
            # med_filt is in stellar rest frame, shift back to observation frame (apply +BERV)
            sp_stellar = np.array([med_filt])
            wave_stellar = np.array([waveref[iord, :]])
            wave_obs = np.array([waveref[iord, :] * mp.relativistic_waveshift(berv)])
            correction = wavecore.wave_to_wave(sp_stellar, wave_stellar, wave_obs)[0]

            if doplot and sample:
                plt.plot(waveref[iord, :], residuals[np.where(g)[0][i]], label='Original Residuals')

            # Subtract median residual template on exposure's native grid
            residuals[np.where(g)[0][i]] -= correction
            # Detrend each exposure by a robust linear model vs wavelength
            try:
                fit = mp.robust_polyfit(waveref[iord, :], residuals[g][i], 1, 3)[0]
                trend = fit[0] * waveref[iord, :] + fit[1]
                residuals[np.where(g)[0][i]] -= trend
            except Exception:
                # In case of failure (e.g. too many NaNs), skip detrending
                pass

            if doplot and sample:
                plt.plot(waveref[iord, :], residuals[np.where(g)[0][i]], label='Corrected Residuals')
                plt.plot(waveref[iord, :], correction + trend, label='Median Correction', color='black', linewidth=2)
                plt.xlabel('Wavelength')
                plt.ylabel('Residuals')
                plt.title(f'Sample Correction for object {uobj}')
                plt.legend()
                plt.show()
                sample = False


    # -------------------------------------------------------------------------
    # Exposure filtering and optional quick looks
    # -------------------------------------------------------------------------
    tbl['DOY'] = tbl['MJDMID'] % 365.24  # Day-of-year proxy (no leap correction)

    # Keep only hot stars and moderate water content to stabilize fits
    keep = tbl['HOTSTAR'] & (tbl['EXPO_H2O'] < 7.0)
    tbl = tbl[keep]
    residuals = residuals[keep, :]

    # Some displays (guarded by doplot)
    vminmax = 0.05
    extent = [np.min(waveref[iord, :]), np.max(waveref[iord, :]), 0, residuals.shape[0]]
    order_airmass = np.argsort(tbl['DRSOBJN'])
    order_h2o = np.argsort(tbl['EXPO_H2O'])

    # Example pixel index for 1D scatter diagnostics
    ipix = len(waveref[iord, :]) // 2

    # Color key for the scatter below (can switch to 'DOY' or others)
    key = 'HUMIDITY'
    if doplot:
        plt.figure(figsize=(8, 6))
        plt.scatter(tbl['EXPO_H2O'], residuals[:, ipix], c=tbl[key], cmap='viridis', s=50)
        plt.xlabel('EXPO_H2O')
        plt.ylabel('RMS Residuals')
        plt.title(f'RMS Residuals vs EXPO_H2O for order {iord}')
        plt.tight_layout()
        cbar = plt.colorbar()
        cbar.set_label(key)
        plt.show()

    # -------------------------------------------------------------------------
    # Per-pixel linear modeling of residuals vs drivers (H2O or AIRMASS)
    # -------------------------------------------------------------------------
    # Vectorized linear fitting for speed (replaces per-pixel loop)
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
    
    # --- Vectorized linear fit for H2O/None pixels ---
    h2o_pix = valid_pix & h2o_mask
    if np.any(h2o_pix):
        x = np.array(tbl['EXPO_H2O'])
        y = residuals[:, h2o_pix]  # shape (nexp, npix_h2o)
        
        # Masked means and covariance (handles NaN)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            x_mean = np.nanmean(x)
            y_mean = np.nanmean(y, axis=0)
            # Broadcast x to match y shape for covariance calculation
            x_centered = x[:, None] - x_mean
            y_centered = y - y_mean[None, :]
            # Mask NaN values
            mask = np.isfinite(y)
            x_centered_masked = np.where(mask, x_centered, 0)
            y_centered_masked = np.where(mask, y_centered, 0)
            n_valid = np.sum(mask, axis=0)
            
            cov_xy = np.sum(x_centered_masked * y_centered_masked, axis=0) / np.maximum(n_valid - 1, 1)
            var_x = np.nanvar(x)
            
            slopes = cov_xy / var_x
            intercepts = y_mean - slopes * x_mean
        
        slope_offset[h2o_pix] = slopes
        dc_offset[h2o_pix] = intercepts
        recon[:, h2o_pix] = intercepts[None, :] + slopes[None, :] * x[:, None]
    
    # --- Vectorized linear fit for O2/CO2/CH4 pixels (morning only) ---
    o2_pix = valid_pix & o2_mask
    if np.any(o2_pix):
        x_full = np.array(tbl['AIRMASS'])
        x = x_full[morning]
        y = residuals[morning, :][:, o2_pix]  # shape (nexp_morning, npix_o2)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            x_mean = np.nanmean(x)
            y_mean = np.nanmean(y, axis=0)
            x_centered = x[:, None] - x_mean
            y_centered = y - y_mean[None, :]
            mask = np.isfinite(y)
            x_centered_masked = np.where(mask, x_centered, 0)
            y_centered_masked = np.where(mask, y_centered, 0)
            n_valid = np.sum(mask, axis=0)
            
            cov_xy = np.sum(x_centered_masked * y_centered_masked, axis=0) / np.maximum(n_valid - 1, 1)
            var_x = np.nanvar(x)
            
            slopes = cov_xy / var_x
            intercepts = y_mean - slopes * x_mean
        
        slope_offset[o2_pix] = slopes
        dc_offset[o2_pix] = intercepts
        # Reconstruct using full airmass (not just morning)
        recon[:, o2_pix] = intercepts[None, :] + slopes[None, :] * x_full[:, None]

    # Save per-order maps
    map_slopes[iord, :] = slope_offset
    map_intercepts[iord, :] = dc_offset

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        n1,p1 = np.nanpercentile(residuals - recon, [16, 84],axis=0)
    rms = (p1 - n1)/2.0
    
    # Compute smoothed RMS envelope for adaptive thresholding
    # Same filtering as for stellar template: medfilt + savgol
    rms_med = medfilt(rms, kernel_size=11)
    rms_envelope = savgol_filter_nan_fast(rms_med, 101, 3, frac_valid=0.3)
    
    map_rms[iord, :] = rms
    map_rms_envelope[iord, :] = rms_envelope

    fits.writeto(outname1, slope_offset, overwrite=True)
    fits.writeto(outname2, dc_offset, overwrite=True)
    fits.writeto(outname3, rms, overwrite=True)
    fits.writeto(outname4, rms_envelope, overwrite=True)
    
    # Paper Figure 4: Per-pixel residual model (only generated once, on first order)
    enabled, output_dir = get_paper_figures_config()
    if enabled and not _paper_figure_done['fig4'] and iord == 0:
        _generate_paper_fig4_residual_model(
            waveref[iord, :], residuals, tbl, slope_offset, dc_offset, 
            main_abso_order, output_dir
        )
        _paper_figure_done['fig4'] = True
    
    # -------------------------------------------------------------------------
    # Diagnostics for this order (guarded by doplot)
    # -------------------------------------------------------------------------
    order_h2o = np.argsort(tbl['EXPO_H2O'])

    if doplot:
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True, squeeze=True)
        ax[0].imshow(residuals[order_h2o], aspect='auto',
                     vmin=-vminmax, vmax=vminmax, extent=extent)
        ax[0].set_xlabel('Pixel')
        ax[0].set_ylabel('Exposure')
        ax[0].set_title(f'Reconstructed Residuals for order {iord} sorted by EXPO_H2O')
        ax[1].imshow(recon[order_h2o], aspect='auto',
                     vmin=-vminmax, vmax=vminmax, extent=extent)
        ax[1].set_xlabel('Pixel')
        ax[1].set_ylabel('Exposure')
        ax[1].set_title(f'Reconstructed Residuals for order {iord} sorted by EXPO_H2O')
        ax[2].imshow(residuals[order_h2o] - recon[order_h2o], aspect='auto',
                     vmin=-vminmax, vmax=vminmax, extent=extent)
        ax[2].set_xlabel('Pixel')
        ax[2].set_ylabel('Exposure')
        ax[2].set_title(f'Residuals - Reconstructed Residuals for order {iord}')
        plt.tight_layout()
        plt.show()

    if doplot:
        fig, ax = plt.subplots(nrows=4, figsize=(10, 8), sharex=True)
        ax[0].plot(waveref[iord, :], dc_offset)
        ax[0].set_ylabel('DC Offset')
        ax[0].set_title(f'DC Offset vs Pixel for order {iord}')
        ax[1].plot(waveref[iord, :], slope_offset)
        ax[1].set_ylabel('Slope Offset')
        ax[1].set_title(f'Slope Offset vs Pixel for order {iord}')
        ax[2].plot(waveref[iord, :], rms)
        ax[2].set_ylabel('RMS Residuals after Model')
        ax[2].set_title(f'RMS Residuals after Model vs Pixel for order {iord}') 
        ax[3].plot(waveref[iord, :], mean_abso[iord, :])
        plt.show()

# -----------------------------------------------------------------------------
# Generate multi-page PDF summary
# -----------------------------------------------------------------------------
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
