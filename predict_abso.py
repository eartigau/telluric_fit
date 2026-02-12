"""
Telluric Absorption Prediction and Correction Pipeline

This module processes spectroscopic data to correct for telluric absorption by:
1. Loading observed spectra and calibration data
2. Modeling atmospheric absorption (H2O, CH4, CO2, O2)
3. Reconstructing and subtracting sky emission (OH airglow)
4. Optimizing absorption exponents
5. Determining stellar radial velocity
6. Generating corrected spectra

The pipeline uses a batch configuration system to process multiple objects
with different parameter sets.

Author: [Your Name]
Date: 2026-01-12
"""

# Early print to show script is starting
from tellu_tools_config import tprint
tprint("Starting predict_abso.py - Loading modules...")

from astropy.io import fits
tprint("  - astropy loaded")
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline as ius
tprint("  - scipy loaded")
from aperocore import math as mp
tprint("  - aperocore.math loaded")
import os
from aperocore.science import wavecore
tprint("  - aperocore.science.wavecore loaded")
import shutil
import warnings
from typing import Dict, Optional, Tuple, List
import argparse
import yaml
import sys
from multiprocessing import Pool
from functools import partial

# Suppress FITS warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*Card is too long.*')
warnings.filterwarnings('ignore', message='.*VerifyWarning.*')

tprint("  - Loading tellu_tools (this may take a moment)...")
import tellu_tools as tt
tprint("  - tellu_tools loaded")
from predict_abso_config import get_batch_config, list_available_objects
from tellu_tools_config import get_user_params
tprint("All modules loaded successfully")

# Paper figure tracking
_paper_figure_done = {'fig6': False}


def get_paper_figures_config(instrument: str = 'NIRPS'):
    """Get paper figures configuration from yaml."""
    config = tt.load_telluric_config()
    paper_config = config.get('paper_figures', {})
    enabled = paper_config.get('enabled', False)
    
    if not enabled:
        return False, None
    
    params = get_user_params(instrument)
    project_path = params['project_path']
    output_dir = os.path.join(project_path, paper_config.get('output_dir', 'paper_figures'))
    os.makedirs(output_dir, exist_ok=True)
    return True, output_dir


def load_template(hdr: fits.Header, template_style: str,
                  project_path: str, instrument: str, obj: str) -> Tuple:
    """
    Load stellar template spectrum and its derivative.

    Parameters
    ----------
    hdr : fits.Header
        FITS header containing stellar parameters
    template_style : str
        'model' for synthetic templates, 'self' for empirical templates
    project_path : str
        Root path to project data
    instrument : str
        Instrument name
    obj : str
        Object name

    Returns
    -------
    spl : InterpolatedUnivariateSpline
        Spline interpolator for template flux
    spl_dv : InterpolatedUnivariateSpline
        Spline interpolator for template velocity gradient
    """
    if template_style == 'model':
        # Fetch synthetic template based on stellar Teff
        spl, spl_dv = tt.fetch_template(hdr)

    elif template_style == 'self':
        # Load empirical template from observations
        template_file = os.path.join(
            project_path,
            f'templates_{instrument}/Template_s1dv_{obj}_sc1d_v_file_A.fits'
        )
        template = Table.read(template_file)
        wave_template = np.array(template['wavelength'])
        flux_template = np.array(template['flux'])

        # Normalize template by low-pass filtered continuum
        flux_template /= mp.lowpassfilter(flux_template, 101)

        # Create spline interpolators (only for finite values)
        g = np.isfinite(flux_template)
        spl = ius(wave_template[g], flux_template[g], k=1, ext=0)

        # Compute velocity gradient for RV determination
        grad = np.gradient(flux_template[g], wave_template[g])
        spl_dv = ius(wave_template[g], grad, k=1, ext=0)

    else:
        raise ValueError(f'Unknown template style: {template_style}')

    return spl, spl_dv


def initialize_residuals(sp: np.ndarray, project_path: str, instrument: str) -> Tuple:
    """
    Load pre-computed residual correction maps.

    These residuals represent systematic variations in the telluric correction
    as a function of absorption strength (used to correct for model imperfections).

    Parameters
    ----------
    sp : np.ndarray
        Spectrum array (shape: n_orders x n_pixels)
    project_path : str
        Root path to project data
    instrument : str
        Instrument name

    Returns
    -------
    residual_intercept : np.ndarray
        Intercept term for each pixel
    residual_slope : np.ndarray
        Slope term (dependence on absorption) for each pixel
    residual_rms : np.ndarray
        RMS of residuals for each pixel
    residual_rms_envelope : np.ndarray
        Smoothed RMS envelope for adaptive thresholding
    """
    residual_intercept = np.zeros_like(sp)
    residual_slope = np.zeros_like(sp)
    residual_rms = np.zeros_like(sp)
    residual_rms_envelope = np.zeros_like(sp)

    for iord in range(sp.shape[0]):
        intercept_file = os.path.join(
            project_path,
            f'residuals_{instrument}/residuals_order_{iord:02d}_intercept.fits'
        )
        slope_file = os.path.join(
            project_path,
            f'residuals_{instrument}/residuals_order_{iord:02d}_slope.fits'
        )
        rms_file = os.path.join(
            project_path,
            f'residuals_{instrument}/residuals_order_{iord:02d}_rms.fits'
        )
        rms_envelope_file = os.path.join(
            project_path,
            f'residuals_{instrument}/residuals_order_{iord:02d}_rms_envelope.fits'
        )

        # Load files if they exist
        if os.path.exists(intercept_file):
            residual_intercept[iord] = fits.getdata(intercept_file)
        if os.path.exists(slope_file):
            residual_slope[iord] = fits.getdata(slope_file)
        if os.path.exists(rms_file):
            residual_rms[iord] = fits.getdata(rms_file)
        if os.path.exists(rms_envelope_file):
            residual_rms_envelope[iord] = fits.getdata(rms_envelope_file)

    return residual_intercept, residual_slope, residual_rms, residual_rms_envelope


def compute_initial_exponents(hdr: fits.Header, model: Dict, hdr_tapas: fits.Header) -> Dict:
    """
    Compute initial absorption exponents from atmospheric parameters.

    Uses empirical models (from compil_stats.py) to predict absorption
    exponents based on airmass, temperature, and pressure.

    Parameters
    ----------
    hdr : fits.Header
        Observation header
    model : dict
        Dictionary of model parameters
    hdr_tapas : fits.Header
        TAPAS reference atmosphere header

    Returns
    -------
    expos : dict
        Dictionary with initial exponents for each molecule
    """
    # Extract atmospheric parameters
    airmass = hdr['AIRMASS']
    temperature = (hdr['TEMPERAT'] + 273.15) / 273.15  # Normalized to 273.15 K
    pressure_norm = hdr['PRESSURE'] / hdr_tapas['PAMBIENT']
    mjd = hdr['MJDMID']

    # O2 absorption (includes seasonal variation, like CO2/CH4)
    params = [model['O2_SLOPE'], model['O2_INTERCEPT'], model['O2_AMP'],
              model['O2_PHASE'], model['O2_AIRMASS_EXP'], model['O2_TEMPERATURE_EXP'],
              model['O2_PRESSURE_EXP']]
    o2_airm = (params[0] * mjd + params[1] +
               params[2] * np.cos(2.0 * np.pi * (mjd % 365.24) / 365.24 + params[3]))
    o2_airm = o2_airm * airmass**params[4] * temperature**params[5] * pressure_norm**params[6]
    expo_o2 = o2_airm * airmass * pressure_norm

    # CO2 absorption (includes seasonal + long-term trend)
    params = [model['CO2_SLOPE'], model['CO2_INTERCEPT'], model['CO2_AMP'],
              model['CO2_PHASE'], model['CO2_AIRMASS_EXP'], model['CO2_TEMPERATURE_EXP'],
              model['CO2_PRESSURE_EXP']]
    co2_abso = (params[0] * mjd + params[1] +
                params[2] * np.cos(2.0 * np.pi * (mjd % 365.24) / 365.24 + params[3]))
    co2_abso = co2_abso * airmass**params[4] * temperature**params[5] * pressure_norm**params[6]
    expo_co2 = (co2_abso / hdr_tapas['VMR_CO2']) * airmass * pressure_norm

    # CH4 absorption (includes seasonal + long-term trend)
    params = [model['CH4_SLOPE'], model['CH4_INTERCEPT'], model['CH4_AMP'],
              model['CH4_PHASE'], model['CH4_AIRMASS_EXP'], model['CH4_TEMPERATURE_EXP'],
              model['CH4_PRESSURE_EXP']]
    ch4_abso = (params[0] * mjd + params[1] +
                params[2] * np.cos(2.0 * np.pi * (mjd % 365.24) / 365.24 + params[3]))
    ch4_abso = ch4_abso * airmass**params[4] * temperature**params[5] * pressure_norm**params[6]
    expo_ch4 = (ch4_abso / hdr_tapas['VMR_CH4']) * airmass * pressure_norm

    return {
        'H2O': None,  # Will be optimized (highly variable)
        'CH4': expo_ch4,
        'CO2': expo_co2,
        'O2': expo_o2
    }


def clean_template_ratio(sp_tmp: np.ndarray, template2: np.ndarray, config: Dict) -> np.ndarray:
    """
    Clean template by removing bad orders and normalizing ratio.

    Parameters
    ----------
    sp_tmp : np.ndarray
        Telluric-corrected spectrum
    template2 : np.ndarray
        Template spectrum
    config : dict
        Configuration with thresholds

    Returns
    -------
    template2 : np.ndarray
        Cleaned template
    bad_orders : list
        List of bad order indices
    """
    bad_orders = []

    for iord in range(sp_tmp.shape[0]):
        ratio = sp_tmp[iord] / template2[iord]

        # Skip orders with too few valid pixels
        if np.mean(np.isfinite(ratio)) < config['min_valid_ratio']:
            ratio = np.nan
            bad_orders.append(iord)
        else:
            # Remove outliers based on ratio thresholds
            median_ratio = np.nanmedian(ratio)
            ratio[np.abs(ratio) > config['template_ratio_threshold_high'] * median_ratio] = np.nan
            ratio[np.abs(ratio) < config['template_ratio_threshold_low'] * median_ratio] = np.nan

            # Smooth the ratio to remove high-frequency noise (only if enough valid points)
            valid_count = np.sum(np.isfinite(ratio))
            if valid_count > config['template_smooth_window'] + 1:
                ratio = mp.lowpassfilter(ratio, config['template_smooth_window'])
            elif valid_count == 1:
                # Too few points to filter
                ratio = np.nan
                bad_orders.append(iord)

        # Apply ratio correction to template
        template2[iord] *= ratio

    # Remove orders with very low flux
    with warnings.catch_warnings(record=True):
        low_flux_order = np.nanmedian(template2, axis=1) < config['low_flux_threshold'] * np.nanmedian(template2)
        template2[low_flux_order, :] = np.nan

    return template2, bad_orders


def apply_post_correction(sp_corr: np.ndarray, abso_scaling: np.ndarray,
                         residual_intercept: np.ndarray, residual_slope: np.ndarray,
                         residual_rms: np.ndarray, residual_rms_envelope: np.ndarray,
                         rms_excess_factor: float,
                         wave: np.ndarray, waveref: np.ndarray) -> Tuple:
    """
    Apply empirical post-correction to spectrum.

    This correction accounts for residual systematics in the telluric model
    that correlate with absorption strength.

    Parameters
    ----------
    sp_corr : np.ndarray
        Corrected spectrum
    abso_scaling : np.ndarray
        Absorption scaling parameter (airmass or H2O exponent)
    residual_intercept : np.ndarray
        Intercept correction map
    residual_slope : np.ndarray
        Slope correction map
    residual_rms : np.ndarray
        RMS of residuals for each pixel
    residual_rms_envelope : np.ndarray
        Smoothed RMS envelope for adaptive thresholding
    rms_excess_factor : float
        Factor for masking (pixels with RMS > factor * envelope are set to NaN)
    wave : np.ndarray
        Wavelength grid
    waveref : np.ndarray
        Reference wavelength grid

    Returns
    -------
    sp_corr : np.ndarray
        Post-corrected spectrum
    post_correction : np.ndarray
        The correction that was applied
    bad_rms_mask : np.ndarray
        Boolean mask of pixels masked due to excessive RMS
    rms_interp : np.ndarray
        RMS interpolated to observation wavelength grid
    rms_threshold : np.ndarray
        Adaptive threshold (envelope * factor)
    """
    # Compute correction in log space
    post_correction_waveref = residual_intercept + residual_slope * abso_scaling
    post_correction_waveref[~np.isfinite(post_correction_waveref)] = 0.0

    # Interpolate to spectrum wavelength grid
    post_correction = wavecore.wave_to_wave(post_correction_waveref, waveref, wave)
    rms_interp = wavecore.wave_to_wave(residual_rms, waveref, wave)
    rms_envelope_interp = wavecore.wave_to_wave(residual_rms_envelope, waveref, wave)

    # Reject extreme corrections (likely unphysical)
    post_correction[np.abs(post_correction) > np.exp(1)] = np.nan

    # Apply correction (multiplicative in linear space = additive in log space)
    sp_corr /= np.exp(post_correction)

    # Mask pixels with unreliable correction (RMS > factor * envelope)
    bad_rms_mask = np.zeros_like(sp_corr, dtype=bool)
    rms_threshold = np.zeros_like(sp_corr)
    if rms_excess_factor > 0:
        rms_threshold = rms_envelope_interp * rms_excess_factor
        bad_rms_mask = rms_interp > rms_threshold
        sp_corr[bad_rms_mask] = np.nan

    return sp_corr, post_correction, bad_rms_mask, rms_interp, rms_threshold


def save_corrected_spectrum(t_name: str, t_outname: str, sp_corr: np.ndarray,
                            abso: np.ndarray, post_correction: np.ndarray,
                            hdr: fits.Header, expos: List[float],
                            molecules: List[str]) -> None:
    """
    Save corrected spectrum to FITS file.

    Parameters
    ----------
    t_name : str
        Input t.fits file path
    t_outname : str
        Output tellupatched_t.fits file path
    sp_corr : np.ndarray
        Corrected spectrum
    abso : np.ndarray
        Absorption model
    post_correction : np.ndarray
        Post-correction applied
    hdr : fits.Header
        Updated header with metadata
    expos : list
        Optimized exponents
    molecules : list
        Molecule names
    """
    # Copy original file
    shutil.copyfile(t_name, t_outname)

    # Compute physical quantities from exponents
    pressure = hdr['PRESSURE']
    pressure0 = hdr['NORMPRES']
    airmass = hdr['AIRMASS']

    # Convert exponents to physical units (zenith, normalized pressure)
    h2ocv = expos[0] * hdr['H2OCV'] / (airmass * pressure / pressure0)
    co2_vmr = expos[1] * hdr['VMR_CO2'] / (airmass * pressure / pressure0)
    ch4_vmr = expos[2] * hdr['VMR_CH4'] / (airmass * pressure / pressure0)
    o2_frac = expos[3] / (airmass * pressure / pressure0)

    # Update header with derived quantities
    hdr['NORMPRES'] = (pressure0, '[kPa] Normalization pressure for TAPAS values')
    hdr['H2O_CV'] = (h2ocv, '[mm] at zenith, normalized pressure')
    hdr['CO2_VMR'] = (co2_vmr, '[ppm] at zenith, normalized pressure')
    hdr['CH4_VMR'] = (ch4_vmr, '[ppm] zenith, normalized pressure')
    hdr['O2_AIRM'] = (o2_frac, 'Airmass equivalent fraction at normalized pressure')

    # Store optimized exponents
    for i, molecule in enumerate(molecules):
        hdr[f'EXPO_{molecule}'] = (expos[i], f'Optimized exponent for {molecule}')

    # Write corrected spectrum and model to file
    with fits.open(t_outname, mode='update') as hdul:
        # FluxA extension: corrected spectrum
        if 'FluxA' in hdul:
            hdul['FluxA'].data = sp_corr
            hdul['FluxA'].header = hdr
        else:
            hdul.append(fits.ImageHDU(data=sp_corr, name='FluxA', header=hdr))

        # Recon extension: reconstruction model (absorption * post-correction)
        if 'Recon' in hdul:
            hdul['Recon'].data = abso * np.exp(post_correction)
        else:
            hdul.append(fits.ImageHDU(data=abso * np.exp(post_correction), name='Recon'))

        hdul.flush()


def process_single_file(file: str, config: Dict, spl, spl_dv,
                       sky_dict: Dict, waveref: np.ndarray,
                       all_abso: np.ndarray, abso_case: np.ndarray,
                       main_abso: np.ndarray, hdr_tapas: fits.Header,
                       model: Dict, blaze: np.ndarray) -> bool:
    """
    Process a single spectroscopic file.

    Parameters
    ----------
    file : str
        Path to input FITS file
    config : dict
        Batch configuration
    spl : InterpolatedUnivariateSpline
        Template flux spline
    spl_dv : InterpolatedUnivariateSpline
        Template velocity gradient spline
    sky_dict : dict
        Sky PCA components
    waveref : np.ndarray
        Reference wavelength grid
    all_abso : np.ndarray
        Pre-computed absorption templates
    abso_case : np.ndarray
        Absorption scaling strategy (per pixel)
    main_abso : np.ndarray
        Main absorber map
    hdr_tapas : fits.Header
        TAPAS reference header
    model : dict
        Model parameters for initial exponents
    blaze : np.ndarray
        Blaze function for weighted optimization

    Returns
    -------
    success : bool
        True if processing succeeded, False otherwise
    """
    project_path = tt.user_params()['project_path']
    instrument = config['instrument']
    obj = config['object']
    batchname = config['batch_name']
    molecules = config['molecules']
    
    # Load telluric config for quality control thresholds
    telluric_config = tt.load_telluric_config()
    rms_excess_factor = telluric_config.get('quality_control', {}).get('rms_excess_factor', 2.0)
    
    # Validate the excess factor
    if rms_excess_factor < 1.5:
        raise ValueError(f'rms_excess_factor ({rms_excess_factor}) must be >= 1.5')

    tprint(f'  Processing {file}')

    # Get file metadata
    try:
        hdr0 = fits.getheader(file)
    except FileNotFoundError:
        tprint(f'  WARNING: FITS file not found: {file}', color='orange')
        tprint(f'  Skipping this file and continuing...', color='orange')
        return False
    
    file_id = hdr0['ARCFILE']

    # Build paths
    t_name = os.path.join(project_path, f'orig_{instrument}', obj,
                          file_id).replace('.fits', 't.fits')

    # Check if t.fits exists
    if not os.path.exists(t_name):
        tprint(f'  Skipping - t.fits does not exist: {t_name}', color='orange')
        return False

    # Output path
    outpath = os.path.join(project_path, f'tellupatched_{instrument}/{obj}_{batchname}/')
    os.makedirs(outpath, exist_ok=True)

    t_outname = t_name.replace('t.fits', 'tellupatched_t.fits').split('/')[-1]
    t_outname = os.path.join(outpath, t_outname)

    # Skip if already processed
    if os.path.exists(t_outname):
        tprint(f'  Skipping - already processed', color='orange')
        return False

    # Load spectrum and wavelength
    hdr = fits.getheader(t_name, ext=1)
    hdr = tt.update_header(hdr)
    sp = fits.getdata(file)
    wavefile = hdr['WAVEFILE']
    wave = fits.getdata(os.path.join(project_path, f'calib_{instrument}/{wavefile}'))

    # Load residual correction maps
    residual_intercept, residual_slope, residual_rms, residual_rms_envelope = initialize_residuals(
        sp, project_path, instrument
    )

    # Extract atmospheric parameters
    hdr['PRESSURE'] = (hdr['HIERARCH ESO TEL AMBI PRES START'] +
                       hdr['HIERARCH ESO TEL AMBI PRES END']) / 2.0
    hdr['AIRMASS'] = (hdr['HIERARCH ESO TEL AIRM START'] +
                      hdr['HIERARCH ESO TEL AIRM END']) / 2.0
    hdr['TEMPERAT'] = hdr['HIERARCH ESO TEL AMBI TEMP']

    # Compute initial exponents from atmospheric model
    expos_dict = compute_initial_exponents(hdr, model, hdr_tapas)
    expos0 = np.array([expos_dict['H2O'], expos_dict['CH4'],
                       expos_dict['CO2'], expos_dict['O2']])
    expos_no_water = np.array([0.0, expos_dict['CH4'],
                               expos_dict['CO2'], expos_dict['O2']])

    msg = f'  Initial exponents: CH4={expos_dict["CH4"]:.3f}, ' \
          f'CO2={expos_dict["CO2"]:.3f}, O2={expos_dict["O2"]:.3f}'
    tprint(msg)

    # Keep original spectrum
    sp0 = np.array(sp)

    # First pass: correct without water to estimate sky
    abso_no_water = tt.construct_abso(wave, expos_no_water, all_abso=all_abso)
    abso_no_water[abso_no_water < tt.user_params()['knee']] = np.nan
    sp_corr_tmp = sp / abso_no_water

    # Estimate sky using low-pass filtered spectrum as continuum
    sp_no_sky = mp.lowpassfilter(sp_corr_tmp.ravel(), config['lowpass_filter_size']).reshape(sp.shape) * abso_no_water
    sky_tmp = tt.sky_pca_fast(wave=wave, spectrum=sp - sp_no_sky, sky_dict=sky_dict,
                              force_positive=True, doplot=tt.user_params()['doplot'])

    # Optimize exponents with sky-subtracted spectrum
    expos = tt.optimize_exponents(wave, sp - sky_tmp, hdr['AIRMASS'], fixed_exponents=expos0, blaze=blaze)
    abso = tt.construct_abso(wave, expos, all_abso=all_abso)
    abso[abso < tt.user_params()['knee']] = np.nan

    # Determine stellar radial velocity
    sp_tmp = (sp - sky_tmp) / abso
    veloshift = tt.get_velo(wave, sp_tmp, spl, dv_amp=config['dv_amp'],
                            doplot=tt.user_params()['doplot'])

    hdr['ABS_VELO'] = (veloshift, 'BERV + systemic velocity (km/s)')
    hdr['SYS_VELO'] = (hdr['ABS_VELO'] - hdr['BERV'], 'Systemic velocity (km/s)')

    # Shift template to stellar rest frame
    template2 = spl(wave * mp.relativistic_waveshift(veloshift))

    # Only keep wavelength domain to be fitted
    template2[wave < np.nanmin(tt.user_params()['wave_fit'])] = np.nan
    template2[wave > np.nanmax(tt.user_params()['wave_fit'])] = np.nan

    # Clean template using spectrum/template ratio
    template2, bad_orders = clean_template_ratio(sp_tmp, template2, config)

    if len(bad_orders) > 0:
        txt_bad = ', '.join([str(b) for b in bad_orders])
        tprint(f'  Warning: bad orders detected: {txt_bad}', color='orange')

    # Reconstruct spectrum model
    sp_recon = template2 * abso

    # Final sky reconstruction with improved model
    sky_recon_final = tt.sky_pca_fast(wave=wave, spectrum=(sp - sp_recon),
                                      sky_dict=sky_dict, force_positive=True,
                                      doplot=tt.user_params()['doplot'])

    # Final optimization of exponents
    with warnings.catch_warnings(record=True):
        expos = tt.optimize_exponents(wave, (sp - sky_recon_final) / template2,
                                     hdr['AIRMASS'], fixed_exponents=expos0, blaze=blaze)

    # Final absorption model with per-molecule masking
    # apply_final_mask=True sets pixels with weight < 0.5 to NaN
    # (based on depth_max, depth_saturated, reject_saturated thresholds from telluric_config.yaml)
    abso = tt.construct_abso(wave, expos, all_abso=all_abso, apply_final_mask=True)
    
    # Store weights for diagnostics before NaN masking
    final_weights = tt.construct_abso.last_weights.copy() if tt.construct_abso.last_weights is not None else None

    # Generate corrected spectrum
    sp_corr = (sp - sky_recon_final) / abso

    # Determine absorption scaling for post-correction
    abso_scaling = np.zeros_like(main_abso)
    abso_scaling[abso_case == 0] = hdr['AIRMASS']
    abso_scaling[abso_case == 1] = expos[0]  # H2O exponent

    # Apply empirical post-correction
    sp_corr, post_correction, bad_rms_mask, rms_interp, rms_threshold = apply_post_correction(
        sp_corr, abso_scaling, residual_intercept, residual_slope, 
        residual_rms, residual_rms_envelope, rms_excess_factor, wave, waveref
    )

    # Reject pixels where sky is brighter than stellar flux
    # sky > (sp0 - sky) means more than half the light is sky
    stellar_flux = sp0 - sky_recon_final
    reject_bright_sky = sky_recon_final > stellar_flux
    sp_corr[reject_bright_sky] = np.nan
    sp[reject_bright_sky] = np.nan

    # Save corrected spectrum
    save_corrected_spectrum(t_name, t_outname, sp_corr, abso, post_correction,
                           hdr, expos, molecules)

    tprint(f'  Wrote corrected spectrum to {os.path.basename(t_outname)}')

    # Diagnostic plot if doplot is enabled
    if config.get('doplot', False):
        # Check if we should skip this plot (sparse sampling)
        skip_counter = config.get('plot_skip_counter', 0)
        if skip_counter > 0:
            config['plot_skip_counter'] = skip_counter - 1
            tprint(f'  Skipping plot ({skip_counter} remaining)', color='cyan')
        else:
            demo_order_config = config['demo_order']
        
            # Handle demo_order as [start, end] range or single value
            if isinstance(demo_order_config, (list, tuple)) and len(demo_order_config) == 2:
                order_range = range(demo_order_config[0], demo_order_config[1] + 1)
            else:
                order_range = [demo_order_config] if isinstance(demo_order_config, int) else [demo_order_config[0]]
        
            # Compute individual molecule transmissions for plotting (use first order as reference)
            mol_colors = {'H2O': 'blue', 'CH4': 'orange', 'CO2': 'green', 'O2': 'red'}
            
            # Get masked pixels for all orders in range (telluric + sky)
            masked_telluric = np.zeros(abso.shape[1], dtype=bool)
            masked_sky = np.zeros(abso.shape[1], dtype=bool)
            for iord in order_range:
                masked_telluric |= ~np.isfinite(abso[iord])
                masked_sky |= reject_bright_sky[iord]
            
            fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
            
            # Helper to shade masked regions with filled rectangles
            def shade_masked(ax, wave_all, mask):
                """Draw grey rectangles for masked regions using midpoint boundaries."""
                if not np.any(mask):
                    return
                
                from matplotlib.patches import Rectangle
                
                ylim = ax.get_ylim()
                ymin, ymax = ylim
                height = ymax - ymin
                
                # Find transitions (edges of masked regions)
                # Pad with False at edges to detect start/end of mask regions
                padded = np.concatenate([[False], mask, [False]])
                diff = np.diff(padded.astype(int))
                starts = np.where(diff == 1)[0]   # Transitions from valid to invalid
                ends = np.where(diff == -1)[0]    # Transitions from invalid to valid
                
                for start_idx, end_idx in zip(starts, ends):
                    # Compute left boundary (midpoint between last valid and first invalid)
                    if start_idx == 0:
                        x_left = wave_all[0]
                    else:
                        x_left = (wave_all[start_idx - 1] + wave_all[start_idx]) / 2.0
                    
                    # Compute right boundary (midpoint between last invalid and first valid)
                    if end_idx >= len(wave_all):
                        x_right = wave_all[-1]
                    else:
                        x_right = (wave_all[end_idx - 1] + wave_all[min(end_idx, len(wave_all)-1)]) / 2.0
                    
                    # Draw rectangle
                    rect = Rectangle((x_left, ymin), x_right - x_left, height,
                                     facecolor='grey', alpha=0.3, edgecolor='none', zorder=0)
                    ax.add_patch(rect)
            
            # Helper to shade sky emission line regions with salmon color
            def shade_emission_lines(ax, wave_all, mask):
                """Draw salmon rectangles for sky emission line regions (sky > stellar)."""
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
                    if start_idx == 0:
                        x_left = wave_all[0]
                    else:
                        x_left = (wave_all[start_idx - 1] + wave_all[start_idx]) / 2.0
                    
                    if end_idx >= len(wave_all):
                        x_right = wave_all[-1]
                    else:
                        x_right = (wave_all[end_idx - 1] + wave_all[min(end_idx, len(wave_all)-1)]) / 2.0
                    
                    rect = Rectangle((x_left, ymin), x_right - x_left, height,
                                     facecolor='lightsalmon', alpha=0.4, edgecolor='none', zorder=0)
                    ax.add_patch(rect)
            
            # Helper to shade excess RMS regions with pink color
            def shade_bad_rms(ax, wave_all, mask):
                """Draw pink rectangles for pixels masked due to excessive RMS."""
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
                    if start_idx == 0:
                        x_left = wave_all[0]
                    else:
                        x_left = (wave_all[start_idx - 1] + wave_all[start_idx]) / 2.0
                    
                    if end_idx >= len(wave_all):
                        x_right = wave_all[-1]
                    else:
                        x_right = (wave_all[end_idx - 1] + wave_all[min(end_idx, len(wave_all)-1)]) / 2.0
                    
                    rect = Rectangle((x_left, ymin), x_right - x_left, height,
                                     facecolor='pink', alpha=0.4, edgecolor='none', zorder=0)
                    ax.add_patch(rect)
            
            # Plot all orders
            sp0_valid_all = []  # Collect VALID original spectra for ylim calculation
            for iord in order_range:
                order_label = f'Order {iord}' if iord == order_range[0] else None
                alpha = 0.7
                
                # Mask for this order (NaN in abso = rejected)
                order_mask = ~np.isfinite(abso[iord])
                
                # Collect valid points for ylim calculation
                sp0_valid = sp0[iord].copy()
                sp0_valid[order_mask] = np.nan
                sp0_valid_all.append(sp0_valid)
                
                # Original spectrum
                axes[0].plot(wave[iord], sp0[iord], 'k-', lw=0.5, alpha=alpha, zorder=2)
                axes[0].plot(wave[iord], sky_recon_final[iord], 'b-', lw=0.5, alpha=0.5, zorder=2)
                
                # Absorption model - show all molecules for each order
                for mol in molecules:
                    mol_trans = all_abso[molecules.index(mol), iord, :] ** expos[molecules.index(mol)]
                    axes[1].plot(wave[iord], mol_trans, '-', lw=0.5, 
                                color=mol_colors[mol], alpha=1.0, zorder=2)
                # Combined: dashed so molecules are visible
                axes[1].plot(wave[iord], abso[iord], 'k--', lw=0.5, alpha=0.7, zorder=3)
                
                # Post-correction and RMS panel
                # Show correction as percentage: (exp(post_correction) - 1) * 100
                corr_pct = (np.exp(post_correction[iord]) - 1) * 100
                axes[2].plot(wave[iord], corr_pct, 'b-', lw=0.5, alpha=alpha, zorder=2)
                axes[2].plot(wave[iord], rms_interp[iord] * 100, 'orange', lw=0.5, alpha=0.7, zorder=2)
                axes[2].plot(wave[iord], rms_threshold[iord] * 100, 'r--', lw=0.5, alpha=0.7, zorder=2)
                
                # Corrected spectrum - masked regions as dotted with low opacity
                sp_corr_masked = np.where(order_mask, sp_corr[iord], np.nan)
                sp_corr_valid = np.where(~order_mask, sp_corr[iord], np.nan)
                axes[3].plot(wave[iord], sp_corr_masked, 'g:', lw=0.5, alpha=0.3, zorder=1)
                axes[3].plot(wave[iord], sp_corr_valid, 'g-', lw=0.5, alpha=alpha, zorder=2)
                axes[3].plot(wave[iord], template2[iord], 'k--', lw=0.5, alpha=0.5, zorder=2)
                
                # Residuals (corrected - template) - also show masked as dotted
                residual = sp_corr[iord] - template2[iord]
                residual_masked = np.where(order_mask, residual, np.nan)
                residual_valid = np.where(~order_mask, residual, np.nan)
                axes[4].plot(wave[iord], residual_masked, 'r:', lw=0.5, alpha=0.3, zorder=1)
                axes[4].plot(wave[iord], residual_valid, 'r-', lw=0.5, alpha=alpha, zorder=2)
            
            # Set ylim for top panel: 0 to 1.5 * 90th percentile of VALID original points
            sp0_valid_flat = np.concatenate(sp0_valid_all)
            ymax_top = 1.5 * np.nanpercentile(sp0_valid_flat, 90)
            axes[0].set_ylim(0, ymax_top)
            
            # Labels and legends (using first order wavelengths for masked shading)
            wave_flat = np.concatenate([wave[iord] for iord in order_range])
            wave_sorted_idx = np.argsort(wave_flat)
            wave_sorted = wave_flat[wave_sorted_idx]
            masked_sorted = np.concatenate([~np.isfinite(abso[iord]) for iord in order_range])[wave_sorted_idx]
            bad_rms_sorted = np.concatenate([bad_rms_mask[iord] for iord in order_range])[wave_sorted_idx]
            
            axes[0].set_ylabel('Flux')
            axes[0].plot([], [], 'k-', lw=0.5, label='Original')
            axes[0].plot([], [], 'b-', lw=0.5, label='Sky')
            axes[0].legend(loc='upper right')
            axes[0].set_title(f'{obj} - Orders {order_range[0]}-{order_range[-1]} - {os.path.basename(file)}')
            # ylim already set above: 0 to 1.5 * 90th percentile
            shade_masked(axes[0], wave_sorted, masked_sorted)
            shade_bad_rms(axes[0], wave_sorted, bad_rms_sorted)
            
            # Add molecule legend (dummy plots)
            for mol in molecules:
                axes[1].plot([], [], '-', color=mol_colors[mol], lw=1, label=mol)
            axes[1].plot([], [], 'k--', lw=1, label='Combined')
            axes[1].set_ylabel('Transmission')
            axes[1].set_ylim(0, 1.1)
            axes[1].legend(loc='lower right', ncol=5, fontsize=8)
            shade_masked(axes[1], wave_sorted, masked_sorted)
            shade_bad_rms(axes[1], wave_sorted, bad_rms_sorted)
            
            # Post-correction panel labels
            axes[2].plot([], [], 'b-', lw=1, label='Correction')
            axes[2].plot([], [], 'orange', lw=1, label='RMS')
            axes[2].plot([], [], 'r--', lw=1, label=f'Threshold ({rms_excess_factor:.1f}x)')
            axes[2].axhline(0, color='grey', lw=0.5, ls='--', zorder=1)
            axes[2].set_ylabel('Correction (%)')
            axes[2].legend(loc='upper right', fontsize=8)
            # Set ylim based on RMS threshold
            rms_thresh_flat = np.concatenate([rms_threshold[iord] * 100 for iord in order_range])
            rms_max = np.nanpercentile(rms_thresh_flat, 95)
            axes[2].set_ylim(-rms_max, rms_max)
            shade_masked(axes[2], wave_sorted, masked_sorted)
            shade_bad_rms(axes[2], wave_sorted, bad_rms_sorted)
            
            axes[3].plot([], [], 'g-', lw=0.5, label='Corrected')
            axes[3].plot([], [], 'k--', lw=0.5, label='Template')
            axes[3].set_ylabel('Flux (corrected)')
            axes[3].legend(loc='upper right')
            
            # Set ylim for corrected plot: 0 to 1.5 * 90th percentile of VALID corrected points only
            sp_corr_valid_list = []
            for iord in order_range:
                order_mask = ~np.isfinite(abso[iord])
                sp_corr_valid_pts = sp_corr[iord].copy()
                sp_corr_valid_pts[order_mask] = np.nan
                sp_corr_valid_list.append(sp_corr_valid_pts)
            sp_corr_valid_flat = np.concatenate(sp_corr_valid_list)
            ymax_bottom = 1.5 * np.nanpercentile(sp_corr_valid_flat, 90)
            axes[3].set_ylim(0, ymax_bottom)
            
            # Shade both telluric-masked and emission line regions
            shade_masked(axes[3], wave_sorted, masked_sorted)
            shade_bad_rms(axes[3], wave_sorted, bad_rms_sorted)
            
            # Also shade sky emission line regions (sky > stellar) with salmon color
            sky_masked_sorted = np.concatenate([reject_bright_sky[iord] for iord in order_range])[wave_sorted_idx]
            shade_emission_lines(axes[3], wave_sorted, sky_masked_sorted)
            
            # Residuals plot (bottom)
            axes[4].axhline(0, color='grey', lw=0.5, ls='--', zorder=1)
            axes[4].set_xlabel('Wavelength (nm)')
            axes[4].set_ylabel('Residual')
            # Set symmetric ylim based on 16-84th percentile of non-masked residuals, 8x wider
            residual_flat = np.concatenate([sp_corr[iord] - template2[iord] for iord in order_range])
            mask_flat = masked_sorted | bad_rms_sorted | np.concatenate([reject_bright_sky[iord] for iord in order_range])[wave_sorted_idx]
            residual_valid = residual_flat[wave_sorted_idx][~mask_flat]
            p16, p84 = np.nanpercentile(residual_valid, [16, 84])
            half_width = (p84 - p16) / 2.0
            axes[4].set_ylim(-8 * half_width, 8 * half_width)
            shade_masked(axes[4], wave_sorted, masked_sorted)
            shade_bad_rms(axes[4], wave_sorted, bad_rms_sorted)
            shade_emission_lines(axes[4], wave_sorted, sky_masked_sorted)
            
            plt.tight_layout()
            
            # Paper Figure 6: Science correction (only generated once)
            global _paper_figure_done
            enabled, output_dir = get_paper_figures_config(instrument)
            if enabled and not _paper_figure_done['fig6']:
                fig_path = os.path.join(output_dir, 'fig6_science_correction.pdf')
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                tprint(f'Paper figure saved: {fig_path}', color='green')
                _paper_figure_done['fig6'] = True
            
            plt.show()
            
            # Prompt user for next plot
            try:
                response = input("Show next spectrum plot? [Y/n/number to skip]: ").strip().lower()
                if response in ['n', 'no']:
                    config['doplot'] = False
                    tprint("Plotting disabled for remaining spectra", color='orange')
                elif response.isdigit() and int(response) > 0:
                    config['plot_skip_counter'] = int(response)
                    tprint(f"Will skip {response} spectra before showing next plot", color='cyan')
            except EOFError:
                # Non-interactive mode, disable plots
                config['doplot'] = False

    return True


def main(batch_name: Optional[str] = None, instrument: Optional[str] = None,
         obj: Optional[str] = None, template_style: Optional[str] = None,
         force_recompute: bool = False):
    """
    Main processing function with batch configuration.

    Parameters
    ----------
    batch_name : str
        Batch identifier
    instrument : str
        Instrument name
    obj : str
        Object name
    template_style : str
        Template style ('model' or 'self')
    force_recompute : bool
        Force recomputation of precomputed absorption grid
    """
    # Load defaults from batch_config.yaml if parameters not provided
    batch_yaml = load_batch_config_yaml()
    
    if batch_name is None:
        if 'batch' in batch_yaml and isinstance(batch_yaml['batch'], dict):
            batch_name = batch_yaml['batch'].get('name', 'skypca_v5')
        else:
            batch_name = batch_yaml.get('batch_name', 'skypca_v5')
    
    if instrument is None:
        instrument = batch_yaml.get('instrument', 'NIRPS')
    
    if template_style is None:
        template_style = batch_yaml.get('template_style', 'model')
    
    if obj is None:
        objects_list = batch_yaml.get('objects', [])
        if objects_list and isinstance(objects_list[0], dict):
            obj = objects_list[0].get('name', 'TOI4552')
        elif objects_list:
            obj = str(objects_list[0])
        else:
            obj = 'TOI4552'
    
    # Load configuration
    config = get_batch_config(batch_name, instrument, obj, template_style)
    tprint(f"{'='*60}")
    tprint(f"TELLURIC CORRECTION PIPELINE")
    tprint(f"{'='*60}")
    tprint(f"Batch: {config['batch_name']}")
    tprint(f"Instrument: {config['instrument']}")
    tprint(f"Object: {config['object']}")
    tprint(f"Template: {config['template_style']}")
    tprint(f"{'='*60}")

    project_path = tt.user_params()['project_path']
    molecules = config['molecules']

    # Load precomputed absorption grid (or compute if not cached)
    tt.precompute_absorption_grid(instrument=instrument, force_recompute=force_recompute)

    # Get file list
    tprint("Searching for files...")
    files = glob.glob(os.path.join(project_path,
                                   f'scidata_{instrument}/{obj}/{instrument}*.fits'))
    
    # Remove dead links (files that don't actually exist)
    files = [f for f in files if os.path.exists(f)]
    
    # Randomize file order if enabled (allows parallel execution of multiple scripts)
    if config.get('randomize_files', True) and len(files) > 1:
        import random
        random.shuffle(files)
        tprint("File order randomized for parallel execution")
    
    tprint(f"Found {len(files)} files to process")

    if len(files) == 0:
        tprint(f"No files found in scidata_{instrument}/{obj}/")
        return

    # Check if all output files already exist
    tprint("Checking for existing output files...")
    batchname = config['batch_name']
    output_dir = os.path.join(project_path, f'tellupatched_{instrument}/{obj}_{batchname}/')
    all_processed = True
    for file in files:
        t_name = file.replace('.fits', '_t.fits')
        t_outname = t_name.replace('t.fits', 'tellupatched_t.fits').split('/')[-1]
        t_outname = os.path.join(output_dir, t_outname)
        if not os.path.exists(t_outname):
            all_processed = False
            break
    
    if all_processed:
        tprint(f"All {len(files)} files already processed. Skipping object...", color='orange')
        return

    # Load reference wavelength grid
    tprint("Loading reference wavelength grid...")
    waveref = tt.getdata_safe(os.path.join(project_path, f'calib_{instrument}/waveref.fits'))

    # Load template
    tprint("Loading stellar template...")
    hdr0 = fits.getheader(files[0])
    spl, spl_dv = load_template(hdr0, template_style, project_path, instrument, obj)

    # Initialize sky PCA dictionary
    tprint("Loading sky PCA components...")
    sky_dict = tt.sky_pca_fast(sky_dict=None)

    # Load blaze function for weighted optimization
    tprint("Loading blaze function...")
    blaze = tt.get_blaze()

    # Load main absorber map and TAPAS reference
    tprint("Loading TAPAS absorption reference...")
    main_abso = fits.getdata(os.path.join(project_path, f'main_absorber_{instrument}.fits'))
    hdr_tapas = fits.getheader(os.path.join(project_path, 'LaSilla_tapas.fits'))

    # Load model parameters for initial exponents
    tprint("Loading model parameters...")
    model_file = f'params_fit_tellu_{instrument}.csv'
    model_table = Table.read(os.path.join(project_path, model_file))
    model = {row['PARAM']: row['VALUE'] for row in model_table}

    # Determine absorption scaling strategy
    # Case 0: use airmass scaling (for O2, minor absorbers)
    # Case 1: use H2O exponent scaling (for H2O-dominated regions)
    abso_case = np.zeros_like(main_abso, dtype=int)
    abso_case[(main_abso == 0) | (main_abso == 4)] = 1

    # Pre-compute absorption templates (for speed)
    tprint("Pre-computing absorption templates...")
    expos_dummy = [1.0, 1.0, 1.0, 1.0]
    all_abso = tt.construct_abso(waveref, expos_dummy, all_abso=None)
    tprint("Initialization complete - starting file processing")

    # Get number of cores from config
    n_cores = config.get('n_cores', 1)
    
    # Process all files
    n_processed = 0
    n_skipped = 0

    if n_cores > 1 and len(files) > 1:
        # Parallel processing
        tprint(f"Using parallel processing with {n_cores} cores", color='cyan')
        
        # Disable plotting for parallel mode (can't do interactive plots)
        config_parallel = config.copy()
        config_parallel['doplot'] = False
        
        # Create argument tuples for each file
        args_list = [
            (file, config_parallel, spl, spl_dv, sky_dict, waveref, all_abso,
             abso_case, main_abso, hdr_tapas, model, blaze)
            for file in files
        ]
        
        # Use starmap for multiple arguments
        with Pool(processes=n_cores) as pool:
            results = pool.starmap(process_single_file, args_list)
        
        n_processed = sum(1 for r in results if r)
        n_skipped = sum(1 for r in results if not r)
        
    else:
        # Serial processing
        for i, file in enumerate(files):
            tprint(f"[{i+1}/{len(files)}] Processing {obj}: {file}")

            success = process_single_file(
                file, config, spl, spl_dv, sky_dict, waveref, all_abso,
                abso_case, main_abso, hdr_tapas, model, blaze
            )

            if success:
                n_processed += 1
            else:
                n_skipped += 1

    # Summary
    tprint(f"{'='*60}")
    tprint(f"PROCESSING COMPLETE")
    tprint(f"{'='*60}")
    tprint(f"Files processed: {n_processed}")
    tprint(f"Files skipped: {n_skipped}")
    tprint(f"Total files: {len(files)}")
    tprint(f"{'='*60}")

    # Generate diagnostic plots if requested
    generate_plots = config.get('generate_plots', False)
    if generate_plots and n_processed > 0:
        tprint("")
        tprint(f"{'='*60}")
        tprint(f"GENERATING INSPECTION PLOTS")
        tprint(f"{'='*60}")

        try:
            import spectrum_inspector

            batch_dir = os.path.join(project_path, f'tellupatched_{instrument}/{obj}_{batchname}/')
            plot_orders = config.get('plot_orders', None)
            order_min = plot_orders[0] if plot_orders else None
            order_max = plot_orders[1] if plot_orders else None

            pdf_paths = spectrum_inspector.inspect_batch(
                batch_dir, instrument=instrument,
                order_min=order_min, order_max=order_max
            )

            tprint(f"Generated {len(pdf_paths)} inspection PDFs")
        except ImportError:
            tprint("Warning: spectrum_inspector module not found, skipping plots", color='orange')
        except Exception as e:
            tprint(f"Warning: Failed to generate plots: {e}", color='orange')



def load_batch_config_yaml(config_path: str = 'batch_config.yaml') -> dict:
    """
    Load batch configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the batch config YAML file
    
    Returns
    -------
    config : dict
        Configuration dictionary with batch settings
    """
    # Try relative to script directory first
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, config_path)
    
    if not os.path.exists(full_path):
        full_path = config_path
    
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            return yaml.safe_load(f)
    
    return {}


if __name__ == '__main__':
    # Load batch_config.yaml to get defaults
    batch_yaml = load_batch_config_yaml()
    
    # Extract defaults from batch_config.yaml
    if 'batch' in batch_yaml and isinstance(batch_yaml['batch'], dict):
        default_batch_name = batch_yaml['batch'].get('name', 'skypca_v5')
    else:
        default_batch_name = batch_yaml.get('batch_name', 'skypca_v5')
    default_instrument = batch_yaml.get('instrument', 'NIRPS')
    default_template = batch_yaml.get('template_style', 'model')
    
    # Get first object from objects list if available
    objects_list = batch_yaml.get('objects', [])
    if objects_list and isinstance(objects_list[0], dict):
        default_object = objects_list[0].get('name', 'TOI4552')
    elif objects_list:
        default_object = str(objects_list[0])
    else:
        default_object = 'TOI4552'
    
    # Command-line interface
    parser = argparse.ArgumentParser(
        description='Telluric absorption correction pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default=None,
                       help='YAML configuration file for batch processing (default: uses batch_config.yaml)')
    parser.add_argument('--batch', type=str, default=default_batch_name,
                       help='Batch name identifier')
    parser.add_argument('--instrument', type=str, default=default_instrument,
                       choices=['NIRPS', 'SPIROU'],
                       help='Instrument name')
    parser.add_argument('--object', type=str, default=default_object,
                       help='Object name')
    parser.add_argument('--template', type=str, default=default_template,
                       choices=['model', 'self'],
                       help='Template style')
    parser.add_argument('--list-objects', action='store_true',
                       help='List available objects and exit')
    parser.add_argument('--recompute', action='store_true',
                       help='Force recomputation of precomputed absorption grid')

    args = parser.parse_args()

    # List objects if requested
    if args.list_objects:
        project_path = tt.user_params()['project_path']
        objects = list_available_objects(args.instrument, project_path)
        tprint(f"Available objects for {args.instrument}:")
        for obj in objects:
            tprint(f"  - {obj}")
    elif args.config:
        # Load from config file
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            tprint(f"ERROR: Config file not found: {args.config}", color='red')
            sys.exit(1)
        except yaml.YAMLError as e:
            tprint(f"ERROR: Failed to parse config file: {e}", color='red')
            sys.exit(1)
        
        # Support both flat 'batch_name' and nested 'batch.name' structure
        if 'batch' in config and isinstance(config['batch'], dict):
            batch_name = config['batch'].get('name', 'skypca_v5')
        else:
            batch_name = config.get('batch_name', 'skypca_v5')
        instrument = config.get('instrument', 'NIRPS')
        template_style = config.get('template_style', 'model')
        objects = config.get('objects', [])
        
        if not objects:
            tprint("ERROR: No objects specified in config file", color='red')
            sys.exit(1)
        
        # Process each object in the config
        for obj_config in objects:
            obj_name = obj_config.get('name')
            obj_template = obj_config.get('template_style', template_style)
            
            tprint(f"Processing {obj_name} with {obj_template} template", color='cyan')
            main(
                batch_name=batch_name,
                instrument=instrument,
                obj=obj_name,
                template_style=obj_template,
                force_recompute=args.recompute
            )
            # Only recompute on first object
            args.recompute = False
    else:
        # Run processing with command-line arguments
        main(
            batch_name=args.batch,
            instrument=args.instrument,
            obj=args.object,
            template_style=args.template,
            force_recompute=args.recompute
        )
