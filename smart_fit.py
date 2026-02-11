"""
Hot Star Telluric Fitting Pipeline

This module processes hot star observations to derive telluric absorption
spectra for use in correcting science observations.

Author: Etienne Artigau
Date: 2026-02
"""

import numpy as np
from astropy.io import fits
import warnings
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import glob
from astropy.coordinates import EarthLocation, get_sun
from astropy.time import Time
from astropy.coordinates import AltAz
import numpy as np
from scipy.signal import savgol_filter
import astropy.units as u
from astropy.time import Time
import numexpr as ne
from aperocore import math as mp
from aperocore.science import wavecore
import sys
import time
from multiprocessing import Pool

# Suppress numpy and astropy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*Card is too long.*')
warnings.filterwarnings('ignore', message='.*VerifyWarning.*')

import tellu_tools as tt
from tellu_tools_config import tprint, get_user_params

MOLECULES = ['H2O', 'CH4', 'CO2', 'O2']
SPEED_OF_LIGHT = 299792.458  # km/s

# Global flag for paper figures (set by main after processing first file)
_paper_figure_done = {'fig1': False, 'fig2': False}


def get_paper_figures_config():
    """Get paper figures configuration from yaml.
    
    Returns
    -------
    enabled : bool
        Whether paper figures are enabled
    output_dir : str or None
        Full path to output directory, or None if disabled
    """
    config = tt.load_telluric_config()
    paper_config = config.get('paper_figures', {})
    enabled = paper_config.get('enabled', False)
    
    if not enabled:
        return False, None
    
    params = get_user_params('NIRPS')  # Gets project_path for current machine
    project_path = params['project_path']
    output_dir = os.path.join(project_path, paper_config.get('output_dir', 'paper_figures'))
    os.makedirs(output_dir, exist_ok=True)
    return True, output_dir


def format_eta(seconds: float) -> str:
    """Format an ETA in seconds into a concise human-readable string."""
    if seconds < 0:
        seconds = 0
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)

    if days:
        return f"{days}d {hours:02d}h {mins:02d}m"
    if hours:
        return f"{hours}h {mins:02d}m"
    if mins:
        if secs:
            return f"{mins}m {secs:02d}s"
        return f"{mins}m"
    return f"{secs}s"


def process_single_hotstar(file: str, outname: str, waveref: np.ndarray,
                           sky_dict: dict, blaze: np.ndarray,
                           instrument: str, doplot: bool = False) -> bool:
    """
    Process a single hot star file to extract telluric absorption.

    Parameters
    ----------
    file : str
        Input FITS file path
    outname : str
        Output FITS file path
    waveref : np.ndarray
        Reference wavelength grid
    sky_dict : dict
        Sky PCA dictionary
    blaze : np.ndarray
        Blaze function
    instrument : str
        Instrument name ('NIRPS' or 'SPIROU')
    doplot : bool
        Whether to show diagnostic plot

    Returns
    -------
    bool
        True if processing succeeded
    """
    try:
        # Load data
        hdr = fits.getheader(file)
        obj_name = hdr.get('OBJECT') or hdr.get('OBJNAME') or 'UNKNOWN'
        
        tprint(f'Processing: {obj_name} -> {os.path.basename(outname)}', color='green')
        
        sp = fits.getdata(file)
        hdr = tt.update_header(hdr)

        wavefile = hdr['WAVEFILE']

        # Get wavefile if missing
        wave_path = f'calib_{instrument}/{wavefile}'
        if not os.path.exists(wave_path):
            if instrument == 'NIRPS':
                cmd = f'scp rali:/cosmos99/nirps/apero-data/nirps_he_online/calib/{wavefile} calib_{instrument}/.'
                tprint(f'Getting {wavefile} from rali...', color='blue')
                os.system(cmd)
            elif instrument == 'SPIROU':
                cmd = f'scp rali:/cosmos99/spirou/apero-data/spirou_offline/calib/{wavefile} calib_{instrument}/.'
                tprint(f'Getting {wavefile} from rali...', color='blue')
                os.system(cmd)

        wave0 = fits.getdata(wave_path)
        # Use observation's wavelength grid (telluric lines imprinted at wave0)
        # Model will be interpolated from waveref to wave0 in construct_abso

        airmass = hdr['AIRMASS']
        pressure = hdr['PRESSURE']  # in kPa
        pressure0 = hdr['NORMPRES']  # in kPa

        # Construct initial absorption model on observation's wavelength grid
        all_abso = tt.construct_abso(wave0, expos=[]*4)

        # Optimize exponents one molecule at a time
        # Start with airmass as initial guess for all molecules
        fixed_exponents = [airmass, airmass, airmass, airmass]
        expo_optimal = np.zeros(4)
        for imolecule in range(4):
            fixed_exponents[imolecule] = None  # Free this molecule
            expo_result = tt.optimize_exponents(wave0, sp, airmass,
                                                fixed_exponents=fixed_exponents,
                                                blaze=blaze)
            expo_optimal[imolecule] = expo_result[imolecule]
            # Fix this molecule to its optimized value before moving to next
            fixed_exponents[imolecule] = expo_optimal[imolecule]

        for molecule, expo in zip(MOLECULES, expo_optimal):
            tprint(f'  Optimized expo for {molecule}: {expo:.4f}', color='blue')

        tprint(f'  Airmass used: {airmass:.4f}', color='blue')

        # Build final transmission model
        trans2 = tt.construct_abso(wave0, expos=expo_optimal, all_abso=all_abso)
        combined_weights = tt.construct_abso.last_weights

        # Compute derived atmospheric parameters
        h2ocv = expo_optimal[0] * hdr['H2OCV'] / (hdr['AIRMASS'] * pressure / pressure0)
        co2_vmr = expo_optimal[1] * hdr['VMR_CO2'] / (hdr['AIRMASS'] * pressure / pressure0)
        ch4_vmr = expo_optimal[2] * hdr['VMR_CH4'] / (hdr['AIRMASS'] * pressure / pressure0)
        o2_frac = expo_optimal[3] / (hdr['AIRMASS'] * pressure / pressure0)

        # Update header
        hdr['NORMPRES'] = pressure0, '[kPa] Normalization pressure for TAPAS values'
        hdr['H2O_CV'] = h2ocv, '[mm] at zenith, normalized pressure'
        hdr['CO2_VMR'] = co2_vmr, '[ppm] at zenith, normalized pressure'
        hdr['CH4_VMR'] = ch4_vmr, '[ppm] zenith, normalized pressure'
        hdr['O2_AIRM'] = o2_frac, 'Airmass equivalent fraction at normalized pressure'

        for i, molecule in enumerate(MOLECULES):
            hdr[f'EXPO_{molecule}'] = expo_optimal[i], f'Optimized exponent for {molecule}'

        # Load processing parameters from config
        config = tt.load_telluric_config()
        medfilt_width_kms = config.get('processing', {}).get('hotstar_medfilt_width', 150)

        # Normalize spectrum per order
        for iord in range(wave0.shape[0]):
            sp[iord] /= np.nanpercentile(sp[iord], 90)

        # Compute telluric-corrected spectrum
        sp_corr = sp / trans2

        # Subtract running median per order (hot stars have no stellar lines)
        # Then fit and subtract sky emission
        sp_median = np.zeros_like(sp_corr)
        for iord in range(wave0.shape[0]):
            # Compute median filter width in pixels from velocity width
            # dv/c = dlambda/lambda -> dpix = width_kms * lambda / (c * dlambda)
            wave_mean = np.nanmean(wave0[iord])
            dlambda = np.nanmedian(np.abs(np.diff(wave0[iord])))
            width_pix = int(medfilt_width_kms * wave_mean / (SPEED_OF_LIGHT * dlambda))
            width_pix = max(width_pix, 5)  # Ensure minimum width
            if width_pix % 2 == 0:
                width_pix += 1  # medfilt requires odd window
            
            # Running median filter
            sp_median[iord] = median_filter(sp_corr[iord], size=width_pix, mode='reflect')

        # Residuals after median subtraction
        residuals = sp_corr - sp_median

        # Create weights for sky fitting: zero where tellurics are rejected
        sky_weights = np.ones_like(sp_corr)
        sky_weights[combined_weights < 0.5] = 0.0

        # Fit sky emission on residuals (in observation wavelength frame)
        # sky_dict is loaded in reference frame but fit is done in observation frame
        sky = tt.sky_pca_fast(wave=wave0, spectrum=residuals * sky_weights,
                              sky_dict=sky_dict, force_positive=True, doplot=False)

        # Final corrected spectrum after sky subtraction
        sp_corr_final = sp_corr - sky

        # Paper figure 2: Sky emission subtraction (only generated once)
        global _paper_figure_done
        enabled, output_dir = get_paper_figures_config()
        if enabled and not _paper_figure_done['fig2']:
            _generate_paper_fig2_sky(wave0, residuals, sky, sp_corr, sp_corr_final, 
                                     combined_weights, obj_name, output_dir)
            _paper_figure_done['fig2'] = True

        # Compute transmission for output
        trans = sp_corr_final / blaze

        # Resample transmission to reference wavelength grid for consistent output
        trans_waveref = wavecore.wave_to_wave(trans, wave0, waveref)

        # Resample combined_weights to reference grid for masking
        weights_waveref = wavecore.wave_to_wave(combined_weights, wave0, waveref)

        # Save log transmission
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            log_trans = np.log(trans_waveref)

        # Mask invalid pixels where telluric correction is unreliable
        log_trans[weights_waveref < 0.5] = np.nan

        # Ensure plain numpy array for FITS compatibility (avoid "Derived must override" error)
        fits.writeto(outname, np.asarray(log_trans), hdr, overwrite=True)

        # Diagnostic plot if requested
        if doplot:
            _plot_hotstar_diagnostic(wave0, sp, trans2, combined_weights, expo_optimal,
                                    all_abso, obj_name, file, instrument, sky, sp_corr_final)

        return True

    except Exception as e:
        tprint(f'Error processing {file}: {e}', color='red')
        return False


def _generate_paper_fig2_sky(wave, residuals, sky, sp_corr, sp_corr_final, 
                             combined_weights, obj_name, output_dir):
    """Generate paper figure 2: Sky emission subtraction.
    
    3-panel figure showing:
    Panel 1: Residuals after telluric correction (before sky subtraction)
    Panel 2: Fitted sky emission model
    Panel 3: Spectrum after sky subtraction
    """
    # Use demo orders from config
    config = tt.load_telluric_config()
    demo_order_config = config.get('demo_order', {}).get('NIRPS', [0, 71])
    
    if isinstance(demo_order_config, (list, tuple)) and len(demo_order_config) == 2:
        order_range = range(demo_order_config[0], demo_order_config[1] + 1)
    else:
        order_range = [demo_order_config] if isinstance(demo_order_config, int) else [demo_order_config[0]]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    for iord in order_range:
        alpha = 0.7
        order_mask = combined_weights[iord] < 0.5
        
        # Plot with masked regions shown transparently
        res_masked = np.where(order_mask, residuals[iord], np.nan)
        res_valid = np.where(~order_mask, residuals[iord], np.nan)
        
        axes[0].plot(wave[iord], res_valid, 'k-', lw=0.5, alpha=alpha)
        axes[0].plot(wave[iord], res_masked, 'k:', lw=0.3, alpha=0.3)
        
        axes[1].plot(wave[iord], sky[iord], 'b-', lw=0.5, alpha=alpha)
        
        after_sky = sp_corr_final[iord].copy()
        after_masked = np.where(order_mask, after_sky, np.nan)
        after_valid = np.where(~order_mask, after_sky, np.nan)
        axes[2].plot(wave[iord], after_valid, 'g-', lw=0.5, alpha=alpha)
        axes[2].plot(wave[iord], after_masked, 'g:', lw=0.3, alpha=0.3)
    
    axes[0].set_ylabel('Residuals (pre-sky)')
    axes[0].set_title(f'Sky Emission Subtraction - {obj_name}')
    
    axes[1].set_ylabel('Sky Model')
    axes[1].set_ylim(bottom=0)
    
    axes[2].axhline(1.0, color='k', ls='--', lw=0.5, alpha=0.5)
    axes[2].set_ylabel('Flux (post-sky)')
    axes[2].set_xlabel('Wavelength (nm)')
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'fig2_sky_subtraction.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    tprint(f'Paper figure saved: {fig_path}', color='green')


def _plot_hotstar_diagnostic(wave, sp, trans2, combined_weights, expo_optimal,
                            all_abso, obj_name, file, instrument, sky, sp_corr_final):
    """Generate 3-panel diagnostic plot for hot star processing.
    
    Panel 1 (top): Sky emission spectrum (to show correction applied)
    Panel 2 (middle): Molecule transmissions and combined transmission
    Panel 3 (bottom): Corrected spectrum after sky subtraction ("après")
    
    Grey shading is based on combined_weights from 3-tier thresholds
    (depth_max, depth_saturated, reject_saturated per molecule).
    
    Parameters
    ----------
    wave : np.ndarray
        Wavelength grid (observation's actual wavelength solution)
    sp : np.ndarray
        Original normalized spectrum
    trans2 : np.ndarray
        Combined transmission model
    combined_weights : np.ndarray
        Weights from construct_abso (< 0.5 = rejected)
    expo_optimal : list
        Optimized exponents for each molecule
    all_abso : np.ndarray
        Base absorption templates
    obj_name : str
        Object name for title
    file : str
        Input file path for title
    instrument : str
        Instrument name
    sky : np.ndarray
        Fitted sky emission spectrum
    sp_corr_final : np.ndarray
        Corrected spectrum after sky subtraction
    """
    # Load demo_order from telluric_config.yaml
    config = tt.load_telluric_config()
    demo_order_config = config.get('demo_order', {}).get(instrument, [0, 71])
    
    if isinstance(demo_order_config, (list, tuple)) and len(demo_order_config) == 2:
        order_range = range(demo_order_config[0], demo_order_config[1] + 1)
    else:
        order_range = [demo_order_config] if isinstance(demo_order_config, int) else [demo_order_config[0]]
    
    mol_colors = {'H2O': 'blue', 'CH4': 'orange', 'CO2': 'green', 'O2': 'red'}
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Helper functions for shading
    def shade_masked(ax, wave_all, mask):
        if not np.any(mask):
            return
        from matplotlib.patches import Rectangle
        ylim = ax.get_ylim()
        ymin, ymax = ylim
        height = ymax - ymin
        padded = np.concatenate([[False], mask, [False]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for start_idx, end_idx in zip(starts, ends):
            x_left = wave_all[0] if start_idx == 0 else (wave_all[start_idx - 1] + wave_all[start_idx]) / 2.0
            x_right = wave_all[-1] if end_idx >= len(wave_all) else (wave_all[end_idx - 1] + wave_all[min(end_idx, len(wave_all)-1)]) / 2.0
            rect = Rectangle((x_left, ymin), x_right - x_left, height,
                             facecolor='grey', alpha=0.3, edgecolor='none', zorder=0)
            ax.add_patch(rect)
    
    # Plot all orders
    sp_all = []  # For ylim calculation based on original spectrum
    sp_corr_final_valid_all = []  # For ylim calculation based on valid points only
    for iord in order_range:
        alpha = 0.7
        
        # Mask for this order (weight < 0.5 = rejected)
        order_mask = combined_weights[iord] < 0.5
        
        # Collect valid sp values for ylim calculation (masking rejected regions)
        sp_valid = sp[iord].copy()
        sp_valid[order_mask] = np.nan
        sp_all.append(sp_valid)
        
        # Collect valid points for final corrected spectrum ylim calculation
        sp_corr_final_valid_pts = sp_corr_final[iord].copy()
        sp_corr_final_valid_pts[order_mask] = np.nan
        sp_corr_final_valid_all.append(sp_corr_final_valid_pts)
        
        # Panel 1: Original spectrum (black) and sky emission (blue) overlaid
        # Sky is shown at its actual level relative to the original spectrum
        axes[0].plot(wave[iord], sp[iord], 'k-', lw=0.5, alpha=alpha, zorder=2)
        axes[0].plot(wave[iord], sky[iord], 'b-', lw=0.5, alpha=0.7, zorder=3)
        
        # Panel 2: Molecule transmissions
        for imol, mol in enumerate(MOLECULES):
            mol_trans = all_abso[imol][iord] ** expo_optimal[imol]
            axes[1].plot(wave[iord], mol_trans, '-', lw=0.5, 
                        color=mol_colors[mol], alpha=0.5, zorder=2)
        # Combined: thinner and more transparent so molecules are visible
        axes[1].plot(wave[iord], trans2[iord], 'k-', lw=0.3, alpha=0.25, zorder=3)
        
        # Panel 3: Corrected spectrum after sky subtraction ("après")
        # Plot masked regions as dotted with low opacity
        sp_final_masked = np.where(order_mask, sp_corr_final[iord], np.nan)
        sp_final_valid = np.where(~order_mask, sp_corr_final[iord], np.nan)
        axes[2].plot(wave[iord], sp_final_masked, 'g:', lw=0.5, alpha=0.3, zorder=1)
        axes[2].plot(wave[iord], sp_final_valid, 'g-', lw=0.5, alpha=alpha, zorder=2)
    
    # Set ylims and labels
    sp_flat = np.concatenate(sp_all)
    ymax_top = 1.5 * np.nanpercentile(sp_flat, 90)
    axes[0].set_ylim(0, ymax_top)
    
    wave_flat = np.concatenate([wave[iord] for iord in order_range])
    wave_sorted_idx = np.argsort(wave_flat)
    wave_sorted = wave_flat[wave_sorted_idx]
    # Mask based on 3-tier thresholds: weight < 0.5 means rejection
    masked_sorted = np.concatenate([(combined_weights[iord] < 0.5) for iord in order_range])[wave_sorted_idx]
    
    axes[0].set_ylabel('Flux + Sky')
    axes[0].set_title(f'{obj_name} - Orders {order_range[0]}-{order_range[-1]} - {os.path.basename(file)}')
    from matplotlib.patches import Patch
    legend_handles = [
        plt.Line2D([], [], color='k', lw=1, label='Spectrum'),
        plt.Line2D([], [], color='b', lw=1, label='Sky emission'),
        Patch(facecolor='grey', alpha=0.3, label='Masked')
    ]
    axes[0].legend(handles=legend_handles, loc='upper right', fontsize=8)
    shade_masked(axes[0], wave_sorted, masked_sorted)
    
    for mol in MOLECULES:
        axes[1].plot([], [], '-', color=mol_colors[mol], lw=1, label=mol)
    axes[1].plot([], [], 'k-', lw=1, label='Combined')
    axes[1].set_ylabel('Transmission')
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(loc='lower right', ncol=5, fontsize=8)
    shade_masked(axes[1], wave_sorted, masked_sorted)
    
    axes[2].axhline(1.0, color='k', ls='--', lw=0.5, alpha=0.5, zorder=2)
    axes[2].plot([], [], 'g-', lw=0.5, label='Corrected')
    axes[2].set_ylabel('Flux (corrected)')
    axes[2].set_xlabel('Wavelength (nm)')
    axes[2].legend(loc='upper right')
    sp_corr_final_valid_flat = np.concatenate(sp_corr_final_valid_all)
    ymax_panel3 = 1.5 * np.nanpercentile(sp_corr_final_valid_flat, 90)
    axes[2].set_ylim(0, ymax_panel3)
    shade_masked(axes[2], wave_sorted, masked_sorted)
    
    plt.tight_layout()
    
    # Save paper figure 1 (hot star transmission) - only once
    global _paper_figure_done
    enabled, output_dir = get_paper_figures_config()
    if enabled and not _paper_figure_done['fig1']:
        fig_path = os.path.join(output_dir, 'fig1_hotstar_transmission.pdf')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        tprint(f'Paper figure saved: {fig_path}', color='green')
        _paper_figure_done['fig1'] = True
    
    plt.show()


def _process_wrapper(args):
    """Wrapper for multiprocessing - unpacks arguments."""
    return process_single_hotstar(*args)


def main(instrument: str = 'NIRPS', doplot: bool = None, n_cores: int = None):
    """
    Main processing function for hot star telluric fitting.

    Parameters
    ----------
    instrument : str
        Instrument name ('NIRPS' or 'SPIROU')
    doplot : bool, optional
        Override doplot from config
    n_cores : int, optional
        Override n_cores from config
    """
    # Load machine config
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), 'batch_config.yaml')
    with open(config_path, 'r') as f:
        batch_config = yaml.safe_load(f)
    
    # Detect machine
    machines = batch_config.get('machines', {})
    machine_config = {}
    for machine_name, mc in machines.items():
        detect_path = mc.get('detect_path', '')
        if os.path.exists(detect_path):
            machine_config = mc
            tprint(f"Detected machine: {machine_name}", color='cyan')
            break
    
    # Get settings (with possible overrides)
    project_path = machine_config.get('project_path', tt.user_params()['project_path'])
    if doplot is None:
        doplot = machine_config.get('doplot', False)
    if n_cores is None:
        n_cores = machine_config.get('n_cores', 1)

    os.chdir(project_path)
    
    tprint(f"{'='*60}")
    tprint(f"HOT STAR TELLURIC FITTING PIPELINE")
    tprint(f"{'='*60}")
    tprint(f"Instrument: {instrument}")
    tprint(f"Project path: {project_path}")
    tprint(f"Cores: {n_cores}")
    tprint(f"Interactive plots: {doplot}")
    tprint(f"{'='*60}")

    # Load shared resources
    tprint("Loading sky PCA components...")
    sky_dict = tt.sky_pca_fast()
    
    tprint("Loading wavelength reference...")
    waveref = fits.getdata(f'calib_{instrument}/waveref.fits')
    
    tprint("Loading blaze function...")
    blaze = tt.get_blaze()

    # Precompute absorption grid for fast optimization
    tprint("Precomputing absorption grid...")
    tt.precompute_absorption_grid(instrument=instrument)

    # Find files
    all_files = np.array(glob.glob(f'hotstars_{instrument}/*pp_e2dsff_*.fits'))

    # Partition into done vs pending
    pending_files = []
    done_files = []
    for file in all_files:
        outname = f'tellu_fit_{instrument}/trans_' + file.split('/')[-1]
        if os.path.exists(outname):
            done_files.append((file, outname))
        else:
            pending_files.append((file, outname))

    N_total = len(all_files)
    N_done = len(done_files)
    N_pending = len(pending_files)

    tprint(f'Hot stars total: {N_total}, already done: {N_done}, to do: {N_pending}', color='cyan')

    if N_pending == 0:
        tprint('All hot stars already processed. Nothing to do.', color='green')
        return

    # Shuffle pending list if randomize_files is enabled (allows parallel execution)
    telluric_config = tt.load_telluric_config()
    randomize_files = telluric_config.get('processing', {}).get('randomize_files', True)
    if randomize_files and len(pending_files) > 1:
        order = np.random.permutation(len(pending_files))
        pending_files = [pending_files[i] for i in order]
        tprint("File order randomized for parallel execution")

    # Process files
    n_processed = 0
    n_skipped = 0
    start_time = time.perf_counter()

    if n_cores > 1 and len(pending_files) > 1:
        # Parallel processing
        tprint(f"Using parallel processing with {n_cores} cores", color='cyan')
        
        # Create argument tuples (no plotting in parallel mode)
        args_list = [
            (file, outname, waveref, sky_dict, blaze, instrument, False)
            for file, outname in pending_files
        ]
        
        with Pool(processes=n_cores) as pool:
            results = pool.map(_process_wrapper, args_list)
        
        n_processed = sum(1 for r in results if r)
        n_skipped = sum(1 for r in results if not r)
        
    else:
        # Serial processing
        durations = []
        plot_skip_counter = 0  # Counter for skipping plots
        for idx, (file, outname) in enumerate(pending_files, start=1):
            loop_start = time.perf_counter()
            
            banner = f"{'*'*20} [{idx}/{N_pending}] {'*'*20}"
            tprint(banner, color='cyan')

            # Check if we should skip this plot (sparse sampling)
            show_plot = doplot
            if plot_skip_counter > 0:
                plot_skip_counter -= 1
                tprint(f'  Skipping plot ({plot_skip_counter} remaining)', color='cyan')
                show_plot = False

            success = process_single_hotstar(
                file, outname, waveref, sky_dict, blaze, instrument, show_plot
            )

            if success:
                n_processed += 1
            else:
                n_skipped += 1

            loop_dur = time.perf_counter() - loop_start
            durations.append(loop_dur)

            remaining = N_pending - idx
            mean_dur = float(np.mean(durations))
            
            tprint(f'Step duration: {loop_dur:0.2f}s | mean {mean_dur:0.2f}s', color='blue')
            if remaining > 0:
                eta_str = format_eta(remaining * mean_dur)
                tprint(f'Remaining {remaining} stars, ETA ~ {eta_str}', color='magenta')

            # Interactive plot prompt
            if show_plot and success:
                try:
                    response = input("Show next plot? [Y/n/number to skip]: ").strip().lower()
                    if response in ['n', 'no']:
                        doplot = False
                        tprint("Plotting disabled for remaining spectra", color='orange')
                    elif response.isdigit() and int(response) > 0:
                        plot_skip_counter = int(response)
                        tprint(f"Will skip {response} spectra before showing next plot", color='cyan')
                except EOFError:
                    doplot = False

    # Summary
    total_time = time.perf_counter() - start_time
    tprint(f"{'='*60}")
    tprint(f"PROCESSING COMPLETE")
    tprint(f"{'='*60}")
    tprint(f"Files processed: {n_processed}")
    tprint(f"Files skipped: {n_skipped}")
    tprint(f"Total time: {format_eta(total_time)}")
    tprint(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hot star telluric fitting pipeline')
    parser.add_argument('--instrument', type=str, default='NIRPS',
                       choices=['NIRPS', 'SPIROU'], help='Instrument name')
    parser.add_argument('--doplot', action='store_true', help='Show diagnostic plots')
    parser.add_argument('--n-cores', type=int, default=None,
                       help='Number of cores (overrides config)')
    
    args = parser.parse_args()
    
    main(instrument=args.instrument, doplot=args.doplot if args.doplot else None,
         n_cores=args.n_cores)
