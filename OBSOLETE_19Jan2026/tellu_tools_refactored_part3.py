"""
Telluric Tools - Part 3: Absorption Modeling and Optimization

This module contains:
- Atmospheric absorption construction
- Exponent optimization
- Variable resolution convolution
- O2 masking

Import this as part of tellu_tools_refactored.
"""

import os
import numpy as np
import warnings
from typing import Dict, Tuple, Optional, List

from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.optimize import minimize
import numexpr as ne

from aperocore import math as mp

from tellu_tools_config import (
    get_user_params,
    get_calib_paths,
    MOLECULES,
    SPEED_OF_LIGHT,
    EXPONENT_OPT_CONFIG,
    CONVOLUTION_CONFIG,
)

# Import from part 1 (will be combined in final version)
from tellu_tools_refactored import E2DS_FWHM, E2DS_EXPO


# ============================================================================
# Convolution with Variable Resolution
# ============================================================================

def super_gauss_fast(xvector: np.ndarray,
                    ew: np.ndarray,
                    expo: np.ndarray) -> np.ndarray:
    """
    Super-Gaussian kernel using numexpr for speed.

    The super-Gaussian is defined as:
        K(x) = exp(-0.5 * |x/ew|^expo)

    where ew = (FWHM/2) / (2*ln(2))^(1/expo)

    Parameters
    ----------
    xvector : np.ndarray
        Input positions (velocity shifts)
    ew : np.ndarray
        Width parameter at each position
    expo : np.ndarray
        Exponent at each position (2 = Gaussian, >2 = super-Gaussian)

    Returns
    -------
    kernel : np.ndarray
        Super-Gaussian profile values
    """
    _ = xvector, expo  # For numexpr scope
    calc_string = 'exp(-0.5*abs(xvector/ew)**expo)'
    return ne.evaluate(calc_string)


def variable_res_conv(wavemap: np.ndarray,
                     spectrum: np.ndarray,
                     res_fwhm: np.ndarray,
                     res_expo: np.ndarray,
                     ker_thres: float = CONVOLUTION_CONFIG['kernel_threshold']
                     ) -> np.ndarray:
    """
    Convolve spectrum with variable resolution kernel.

    This function accounts for the varying spectral resolution across
    the detector by convolving with position-dependent kernels.

    Parameters
    ----------
    wavemap : np.ndarray
        Wavelength grid
    spectrum : np.ndarray
        Spectrum to convolve
    res_fwhm : np.ndarray
        FWHM of spectral resolution (same shape as spectrum)
    res_expo : np.ndarray
        PSF exponent (same shape as spectrum)
    ker_thres : float
        Kernel amplitude threshold to stop convolution

    Returns
    -------
    spectrum2 : np.ndarray
        Convolved spectrum

    Notes
    -----
    The convolution is performed in velocity space to properly account
    for the logarithmic wavelength dependence of the resolution.

    This is critical for telluric modeling as the atmospheric absorption
    lines must be convolved to the instrumental resolution.
    """
    shape0 = spectrum.shape

    # Flatten if 2D
    if len(shape0) == 2:
        res_fwhm = res_fwhm.ravel()
        res_expo = res_expo.ravel()
        wavemap = wavemap.ravel()
        spectrum = spectrum.ravel()

    # Initialize output
    sumker = np.zeros_like(spectrum)
    spectrum2 = np.zeros_like(spectrum)

    # Determine scan range
    scale1 = np.max(res_fwhm)
    scale2 = np.median(np.gradient(wavemap) / wavemap) * SPEED_OF_LIGHT
    range_scan = CONVOLUTION_CONFIG['range_scan_scale'] * (scale1 / scale2)
    range_scan = int(np.ceil(range_scan))

    # Mask NaN pixels
    valid_pix = np.isfinite(spectrum)
    spectrum[~valid_pix] = 0.0
    valid_pix = valid_pix.astype(float)

    # Sort offsets by distance from center
    range2 = np.arange(-range_scan, range_scan)
    range2 = range2[np.argsort(abs(range2))]

    # Super-Gaussian width parameter
    ew = (res_fwhm / 2) / (2 * np.log(2))**(1 / res_expo)

    # Convolve by scanning through offsets
    for offset in range2:
        # Velocity shift for this offset
        dv = SPEED_OF_LIGHT * (wavemap / np.roll(wavemap, offset) - 1)

        # Compute kernel
        ker = super_gauss_fast(dv, ew, res_expo)

        # Stop if kernel too small
        if np.max(ker) < ker_thres:
            break

        # No weight for NaN pixels
        ker = ker * valid_pix

        # Add to convolution
        spectrum2 += np.roll(spectrum, offset) * ker
        sumker += ker

    # Normalize
    with warnings.catch_warnings(record=True):
        spectrum2 = spectrum2 / sumker

    # Reshape if needed
    if len(shape0) == 2:
        spectrum2 = spectrum2.reshape(shape0)

    return spectrum2


# ============================================================================
# Atmospheric Absorption Construction
# ============================================================================

def construct_abso(wave: np.ndarray,
                  expos: List[float],
                  all_abso: Optional[np.ndarray] = None,
                  instrument: str = 'NIRPS') -> np.ndarray:
    """
    Construct atmospheric absorption model.

    Computes transmission T = Π_i T_i^expos[i] where T_i is the
    transmission of molecule i and expos[i] is its scaling exponent.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength grid (n_orders, n_pixels)
    expos : list of float
        Exponents for [H2O, CH4, CO2, O2]
    all_abso : np.ndarray, optional
        Pre-computed absorption templates (shape: 4, n_orders, n_pixels).
        If None, loads and computes from TAPAS file.
    instrument : str
        Instrument name

    Returns
    -------
    trans : np.ndarray
        Total atmospheric transmission (same shape as wave)

    Notes
    -----
    The exponents scale the optical depth:
        τ_i = expos[i] * τ_i,TAPAS

    This allows modeling variations in column densities and airmass.

    The final transmission is convolved with the instrumental resolution.
    """
    # Load molecular absorptions if not provided
    if all_abso is None:
        params = get_user_params(instrument)

        if instrument == 'NIRPS':
            transm_file = '../LaSilla_NIRPS_tapas.fits'
        elif instrument == 'SPIROU':
            transm_file = '../MaunaKea_tapas.fits'
        else:
            raise ValueError(f'Unknown instrument: {instrument}')

        transm_file = os.path.join(params['project_path'], transm_file)
        transm_table = Table.read(transm_file)

        # Create table with relevant molecules
        tbl = Table()
        tbl['wavelength'] = transm_table['wavelength']
        for mol in MOLECULES:
            tbl[mol] = transm_table[mol]

        # Filter wavelength range
        keep_wave = ((tbl['wavelength'] >= np.min(wave)) &
                    (tbl['wavelength'] <= np.max(wave)))
        tbl = tbl[keep_wave]

        transm_wave = np.array(tbl['wavelength'])
        molecules = np.array(tbl.keys())[1:]

        # Interpolate onto observation grid
        all_abso = np.zeros((len(molecules), wave.shape[0], wave.shape[1]))
        for i, molecule in enumerate(molecules):
            all_abso[i] = ius(transm_wave, tbl[molecule], ext=0, k=1)(wave)
            all_abso[i][all_abso[i] < 0] = 0.0

        return all_abso

    # Construct combined absorption using numexpr (fast)
    absos = np.ones_like(wave)

    # Build expression dynamically
    expr_parts = [f'a{i}**e{i}' for i in range(len(MOLECULES))]
    expr = 'absos * ' + ' * '.join(expr_parts)

    # Build local dictionary for numexpr
    local_dict = {'absos': absos}
    for i in range(len(MOLECULES)):
        local_dict[f'a{i}'] = all_abso[i]
        local_dict[f'e{i}'] = expos[i]

    # Evaluate (equivalent to: absos *= all_abso[i]**expos[i] for all i)
    absos = ne.evaluate(expr, local_dict=local_dict)

    # Convolve with instrumental resolution
    trans2 = variable_res_conv(wave, absos, E2DS_FWHM, E2DS_EXPO)

    # Mask very low transmission
    knee = get_user_params(instrument)['knee']
    trans2[trans2 < knee / 5.0] = np.nan

    return trans2


# ============================================================================
# Exponent Optimization
# ============================================================================

def optimize_exponents(wave: np.ndarray,
                      sp: np.ndarray,
                      airmass: float,
                      fixed_exponents: Optional[List] = None,
                      guess: Optional[List] = None,
                      knee: float = 0.3,
                      instrument: str = 'NIRPS') -> List[float]:
    """
    Optimize atmospheric absorption exponents.

    Finds the best-fit exponents by minimizing the gradient of the
    telluric-corrected spectrum (weighted by transmission).

    Parameters
    ----------
    wave : np.ndarray
        Wavelength grid
    sp : np.ndarray
        Observed spectrum
    airmass : float
        Observation airmass
    fixed_exponents : list of 4, optional
        List with None for free exponents, float for fixed ones.
        Order: [H2O, CH4, CO2, O2]
    guess : list of 4, optional
        Initial guess for exponents
    knee : float
        Absorption threshold for masking
    instrument : str
        Instrument name

    Returns
    -------
    expo_optimal : list of float
        Optimized exponents [H2O, CH4, CO2, O2]

    Notes
    -----
    The optimization minimizes:
        σ(∇(sp/T) * w(T) * T)

    where:
    - T is transmission
    - w(T) is weighting function (down-weights strong absorption)
    - ∇ is spatial gradient

    The gradient should be flat if the telluric correction is perfect.
    """
    # Build initial exponents
    if fixed_exponents is None:
        expos_input = [airmass] * 4
    else:
        expos_input = [airmass if fe is None else fe for fe in fixed_exponents]

    # Pre-compute absorption reference
    all_abso = construct_abso(wave, expos=expos_input, instrument=instrument)
    abso0 = np.nanprod(all_abso, axis=0)

    trans_ref = construct_abso(wave, expos=expos_input, all_abso=all_abso,
                               instrument=instrument)

    # Pre-compute masks and weights
    relevant = ((trans_ref >= knee * 0.5) & (trans_ref <= 0.95))

    from tellu_tools_refactored_part2 import weight_fall
    ww = weight_fall(trans_ref, knee=knee)

    # Wavelength mask for fitting region
    params = get_user_params(instrument)
    wave_fit = params['wave_fit']
    wave_mask_inv = ~((wave >= wave_fit[0]) & (wave <= wave_fit[1]) & (abso0 > knee / 5.0))

    # Pre-compute pixel-to-pixel RMS (vectorized)
    pix2pixrms = np.nanmedian(np.abs(np.diff(sp, axis=1)), axis=1)
    pix2pixrms_30 = 30.0 * pix2pixrms[:, np.newaxis]

    # Build indices for variable exponents
    if fixed_exponents is None:
        var_indices = [0, 1, 2, 3]
    else:
        var_indices = [i for i in range(4) if fixed_exponents[i] is None]

    print('Starting exponent optimization...')

    def optimize_expo(variable_expos):
        """Objective function to minimize."""
        # Reconstruct full exponents list
        if fixed_exponents is None:
            expos = list(variable_expos)
        else:
            expos = list(fixed_exponents)
            for j, i in enumerate(var_indices):
                expos[i] = variable_expos[j]

        # Compute transmission model
        trans2 = construct_abso(wave, expos=expos, all_abso=all_abso,
                               instrument=instrument)
        trans2[trans2 < knee] = np.nan

        # Corrected spectrum
        corr = sp / trans2
        corr[wave_mask_inv] = np.nan

        # Gradient weighted by transmission
        grad = np.gradient(corr, axis=1) * weight_fall(trans2, knee=knee) * trans2

        # Objective: standard deviation of weighted gradient
        val_sum = np.nanstd(grad)

        # Progress output
        strout = ' '.join(f'{mol}: {exp:.4f}' for mol, exp in zip(MOLECULES, expos))
        print(f'\r{strout} | Gradient STD: {val_sum:.8f}', end='', flush=True)

        return val_sum

    # Build initial guess and bounds
    config = EXPONENT_OPT_CONFIG

    if guess is None:
        x0_full = [1.0, airmass, airmass, airmass]
        bounds_full = [
            config['h2o_bounds'],
            (airmass * (1 - config['airmass_tolerance']),
             airmass * (1 + config['airmass_tolerance'])),
            (airmass * (1 - config['airmass_tolerance']),
             airmass * (1 + config['airmass_tolerance'])),
            (airmass * (1 - config['airmass_tolerance']),
             airmass * (1 + config['airmass_tolerance'])),
        ]
    else:
        x0_full = []
        bounds_full = []
        for i in range(4):
            if guess[i] is None:
                x0_full.append(1.0 if i == 0 else airmass)
                if i == 0:
                    bounds_full.append(config['h2o_bounds'])
                else:
                    bounds_full.append((airmass * 0.9, airmass * 1.1))
            else:
                x0_full.append(guess[i])
                bounds_full.append((guess[i] * 0.9, guess[i] * 1.1))

    # Extract only variable parameters
    if fixed_exponents is None:
        x0 = x0_full
        bounds = bounds_full
    else:
        x0 = [x0_full[i] for i in var_indices]
        bounds = [bounds_full[i] for i in var_indices]

    # Optimize
    with np.errstate(invalid='ignore'):
        result = minimize(
            optimize_expo,
            x0=x0,
            bounds=bounds,
            method=config['method'],
            tol=config['tolerance']
        )

    print()  # Newline after progress

    # Reconstruct full exponents list
    if fixed_exponents is None:
        expo_optimal = list(result.x)
    else:
        expo_optimal = list(fixed_exponents)
        for j, i in enumerate(var_indices):
            expo_optimal[i] = result.x[j]

    return expo_optimal


# ============================================================================
# O2 Masking
# ============================================================================

def mask_o2(wave: np.ndarray, instrument: str = 'NIRPS') -> np.ndarray:
    """
    Create mask for O2 absorption lines.

    O2 lines are particularly difficult to model accurately due to
    their temperature dependence and saturation effects. This function
    identifies and masks strong O2 features.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength grid
    instrument : str
        Instrument name

    Returns
    -------
    mask : np.ndarray (bool)
        True where O2 absorption should be masked

    Notes
    -----
    The mask includes a velocity window around each identified O2 line
    to account for stellar RV variations.
    """
    params = get_user_params(instrument)

    if instrument == 'NIRPS':
        transm_file = '../LaSilla_NIRPS_tapas.fits'
    elif instrument == 'SPIROU':
        transm_file = '../MaunaKea_tapas.fits'
    else:
        raise ValueError(f'Unknown instrument: {instrument}')

    transm_file = os.path.join(params['project_path'], transm_file)
    transm_table = Table.read(transm_file)

    trans_o2 = np.array(transm_table['O2'])

    # Identify O2 line centers (local minima in transmission)
    lines = np.where(
        (np.gradient(np.gradient(trans_o2)) > 0) &
        (trans_o2 < 0.8) &
        (trans_o2 < np.roll(trans_o2, 1)) &
        (trans_o2 < np.roll(trans_o2, -1))
    )

    # Create mask
    mask = np.zeros_like(transm_table['wavelength'], dtype=bool)

    # Velocity step in m/s
    velostep = (np.nanmedian(np.gradient(transm_table['wavelength']) /
                             transm_table['wavelength'] * 3e5) * 1000)

    window_size = 8000  # m/s, velocity window around lines

    # Mask region around each line
    for dv in range(-int(window_size / velostep), int(window_size / velostep) + 1):
        mask[lines[0] + dv] = True

    # Interpolate mask onto observation grid
    spl_o2 = ius(transm_table['wavelength'], mask.astype(float), ext=0, k=1)

    return spl_o2(wave) > 0.5


# Export commonly used functions
__all__ = [
    'super_gauss_fast',
    'variable_res_conv',
    'construct_abso',
    'optimize_exponents',
    'mask_o2',
]
