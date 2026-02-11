"""
Telluric Correction Tools - Refactored Version

This module provides utilities for telluric absorption correction in
astronomical spectroscopy. It includes functions for:
- Sky emission reconstruction using PCA
- Atmospheric absorption modeling
- Stellar template handling
- Velocity determination
- Header management
- Calibration data loading

Key improvements over original:
- Modular configuration system
- Comprehensive documentation
- Cleaner code structure
- Better error handling
- Consistent naming conventions

Author: Refactored from tellu_tools.py
Date: 2026-01-12
"""

# Standard library
import os
import warnings
from typing import Dict, Optional, Tuple, List, Union, Any

# Third-party imports
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
from tqdm import tqdm

# Astropy
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
import astropy.units as u

# Scipy
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.signal import savgol_filter

# YAML support
import yaml

# APERO
from aperocore import math as mp
from aperocore.science import wavecore

# Local imports
from tellu_tools_config import (
    get_user_params,
    get_calib_paths,
    MOLECULES,
    SPEED_OF_LIGHT,
    SKY_PCA_CONFIG,
    VELOCITY_CONFIG,
    EXPONENT_OPT_CONFIG,
    TEMPLATE_CONFIG,
    AIRMASS_CONFIG,
    CONVOLUTION_CONFIG,
    HEADER_KEYS,
    PROCESSING_DEFAULTS,
    validate_instrument,
    tprint,
)

# Default path to telluric config
TELLURIC_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'telluric_config.yaml')


# ============================================================================
# Module-level Configuration and Calibration Loading
# ============================================================================

# Default instrument (can be overridden)
DEFAULT_INSTRUMENT = 'NIRPS'

def _load_instrument_calibration(instrument: str = DEFAULT_INSTRUMENT) -> Tuple:
    """
    Load instrument calibration data (FWHM, EXPO, blaze).

    This is called once at module load time to cache calibration data.

    Parameters
    ----------
    instrument : str
        Instrument name

    Returns
    -------
    E2DS_FWHM : np.ndarray
        Spectral resolution FWHM map
    E2DS_EXPO : np.ndarray
        Spectral PSF exponent map
    blaze : np.ndarray
        Blaze function
    """
    validate_instrument(instrument)

    params = get_user_params(instrument)
    calib_paths = get_calib_paths(instrument, params['project_path'])

    # Load FWHM and EXPO
    E2DS_FWHM = fits.getdata(calib_paths['fwhm_file'], calib_paths['fwhm_ext'])
    E2DS_EXPO = fits.getdata(calib_paths['fwhm_file'], calib_paths['expo_ext'])

    # Load and normalize blaze
    blaze = fits.getdata(calib_paths['blaze_file'])
    for iord in range(blaze.shape[0]):
        blaze[iord] /= np.nanpercentile(blaze[iord], 90)

    return E2DS_FWHM, E2DS_EXPO, blaze


# Load calibration for default instrument
E2DS_FWHM, E2DS_EXPO, BLAZE = _load_instrument_calibration(DEFAULT_INSTRUMENT)


# ============================================================================
# Configuration Access Functions (Backward Compatibility)
# ============================================================================

def user_params(instrument: str = DEFAULT_INSTRUMENT) -> Dict[str, Any]:
    """
    Get user parameters for the specified instrument.

    Parameters
    ----------
    instrument : str
        Instrument name ('NIRPS' or 'SPIROU')

    Returns
    -------
    params : dict
        Configuration parameters
    """
    return get_user_params(instrument)


def get_blaze() -> np.ndarray:
    """
    Get the blaze function for the current instrument.

    Returns
    -------
    blaze : np.ndarray
        Normalized blaze function
    """
    return BLAZE


def get_valid_transmission_mask(transmission: np.ndarray,
                                 depth_max: float = 0.5,
                                 depth_saturated: float = 0.2,
                                 reject_saturated: float = 0.8) -> np.ndarray:
    """
    Create a boolean mask of valid pixels based on telluric absorption depth.

    Pixels are marked invalid (False) if:
    1. Transmission is below depth_max (too deep absorption)
    2. Adjacent to a saturated pixel (transmission < depth_saturated) and
       transmission is below reject_saturated

    This handles the case where a 50% transmission pixel may be unreliable
    if it's next to a heavily saturated (e.g., 1% transmission) pixel.

    Parameters
    ----------
    transmission : np.ndarray
        1D array of transmission values (0 to 1)
    depth_max : float, optional
        Maximum transmission below which pixels are invalid. Default: 0.5
    depth_saturated : float, optional
        Transmission threshold for saturated pixels. Default: 0.2
    reject_saturated : float, optional
        Rejection threshold around saturated pixels. Default: 0.8

    Returns
    -------
    valid_mask : np.ndarray
        Boolean array where True = valid pixel, False = invalid pixel

    Examples
    --------
    >>> trans = np.array([0.9, 0.8, 0.7, 0.45, 0.7, 0.9])
    >>> get_valid_transmission_mask(trans, 0.5, 0.2, 0.8)
    array([ True,  True,  True, False,  True,  True])

    >>> trans = np.array([0.9, 0.75, 0.1, 0.6, 0.9])
    >>> get_valid_transmission_mask(trans, 0.5, 0.2, 0.8)
    array([ True, False, False, False,  True])

    Raises
    ------
    ValueError
        If depth parameters are not in correct order:
        depth_saturated < depth_max < reject_saturated
    """
    # Validate parameter ordering
    if not (depth_saturated < depth_max):
        raise ValueError(
            f"Invalid depth parameters: depth_saturated ({depth_saturated}) must be LESS than "
            f"depth_max ({depth_max}).\n"
            f"  - depth_saturated: transmission below which pixels are considered saturated "
            f"(very deep absorption, e.g., 0.2 = 80% absorbed)\n"
            f"  - depth_max: transmission below which pixels are rejected outright "
            f"(moderate absorption, e.g., 0.5 = 50% absorbed)\n"
            f"Expected: depth_saturated < depth_max (e.g., 0.2 < 0.5)"
        )

    if not (depth_max < reject_saturated):
        raise ValueError(
            f"Invalid depth parameters: depth_max ({depth_max}) must be LESS than "
            f"reject_saturated ({reject_saturated}).\n"
            f"  - depth_max: transmission below which pixels are rejected outright "
            f"(e.g., 0.5 = 50% absorbed)\n"
            f"  - reject_saturated: transmission threshold for the extended rejection zone "
            f"around saturated pixels (e.g., 0.8 = only 20% absorbed but near saturation)\n"
            f"Expected: depth_max < reject_saturated (e.g., 0.5 < 0.8)"
        )

    transmission = np.asarray(transmission)
    n = len(transmission)

    # Start with all pixels valid
    valid_mask = np.ones(n, dtype=bool)

    # Rule 1: Mark pixels below depth_max as invalid
    valid_mask[transmission < depth_max] = False

    # Rule 2: Find saturated pixels and expand rejection zone
    saturated_indices = np.where(transmission < depth_saturated)[0]

    for idx in saturated_indices:
        # Expand left from saturated pixel
        for i in range(idx - 1, -1, -1):
            if transmission[i] >= reject_saturated:
                break
            valid_mask[i] = False

        # Expand right from saturated pixel
        for i in range(idx + 1, n):
            if transmission[i] >= reject_saturated:
                break
            valid_mask[i] = False

    return valid_mask


def get_transmission_weights(transmission: np.ndarray,
                              depth_max: float = 0.5,
                              depth_saturated: float = 0.2,
                              reject_saturated: float = 0.8,
                              transition_sigma: Optional[float] = None) -> np.ndarray:
    """
    Compute smooth weights for pixels based on telluric absorption depth.

    Unlike get_valid_transmission_mask which returns binary True/False,
    this returns fractional weights (0 to 1) with smooth transitions.
    This prevents convergence issues in optimization loops where binary
    masking can cause oscillations.

    Weights are reduced (toward 0) if:
    1. Transmission approaches depth_max (smooth sigmoid transition)
    2. Pixel is near a saturated region (transmission < depth_saturated),
       with weight decaying smoothly based on transmission relative to
       reject_saturated

    Parameters
    ----------
    transmission : np.ndarray
        1D array of transmission values (0 to 1)
    depth_max : float, optional
        Transmission threshold for weight reduction. Default: 0.5
    depth_saturated : float, optional
        Transmission threshold for saturated pixels. Default: 0.2
    reject_saturated : float, optional
        Transmission threshold for contamination from saturated pixels. Default: 0.8
    transition_sigma : float, optional
        Sigma for sigmoid transition (smaller = sharper). If None, loads from
        telluric_config.yaml. Default in config: 0.02

    Returns
    -------
    weights : np.ndarray
        Float array with values 0 to 1, where 1 = full weight, 0 = rejected

    Notes
    -----
    The weight is computed as:
        w = w_depth * w_contamination

    where:
    - w_depth: sigmoid centered at depth_max
    - w_contamination: accounts for proximity to saturated pixels

    Examples
    --------
    >>> trans = np.array([0.9, 0.8, 0.7, 0.45, 0.7, 0.9])
    >>> weights = get_transmission_weights(trans, 0.5, 0.2, 0.8)
    >>> # weights near 1 for high transmission, near 0 for low
    """
    # Validate parameter ordering
    if not (depth_saturated < depth_max):
        raise ValueError(
            f"Invalid depth parameters: depth_saturated ({depth_saturated}) must be LESS than "
            f"depth_max ({depth_max}).\n"
            f"  - depth_saturated: transmission below which pixels are considered saturated "
            f"(very deep absorption, e.g., 0.2 = 80% absorbed)\n"
            f"  - depth_max: transmission below which pixels are rejected outright "
            f"(moderate absorption, e.g., 0.5 = 50% absorbed)\n"
            f"Expected: depth_saturated < depth_max (e.g., 0.2 < 0.5)"
        )

    if not (depth_max < reject_saturated):
        raise ValueError(
            f"Invalid depth parameters: depth_max ({depth_max}) must be LESS than "
            f"reject_saturated ({reject_saturated}).\n"
            f"  - depth_max: transmission below which pixels are rejected outright "
            f"(e.g., 0.5 = 50% absorbed)\n"
            f"  - reject_saturated: transmission threshold for the extended rejection zone "
            f"around saturated pixels (e.g., 0.8 = only 20% absorbed but near saturation)\n"
            f"Expected: depth_max < reject_saturated (e.g., 0.5 < 0.8)"
        )

    # Load transition_sigma from config if not provided
    if transition_sigma is None:
        config = load_telluric_config()
        transition_sigma = config.get('weighting', {}).get('transition_sigma', 0.02)

    transmission = np.asarray(transmission, dtype=np.float64)
    n = len(transmission)

    # Weight 1: Smooth sigmoid for depth_max threshold
    # Weight goes from ~0 (below depth_max) to ~1 (above depth_max)
    # Using transition_sigma directly as the sigmoid width
    w_depth = 1.0 / (1.0 + np.exp(-(transmission - depth_max) / transition_sigma))

    # Weight 2: Contamination from saturated pixels
    # Saturated pixels (trans < depth_saturated) contaminate neighbors
    # Contamination extends outward until trans >= reject_saturated
    # ALL pixels in this zone get contamination = 1 (weight = 0)
    
    is_saturated = transmission < depth_saturated
    is_above_reject = transmission >= reject_saturated
    
    contamination = np.zeros(n, dtype=np.float64)

    # Forward pass: propagate contamination rightward
    in_contamination_zone = False
    for i in range(n):
        if is_saturated[i]:
            # Enter/stay in contamination zone
            in_contamination_zone = True
            contamination[i] = 1.0
        elif in_contamination_zone:
            if is_above_reject[i]:
                # Exit contamination zone - transmission is high enough
                in_contamination_zone = False
            else:
                # Still in contamination zone
                contamination[i] = 1.0

    # Backward pass: propagate contamination leftward
    in_contamination_zone = False
    for i in range(n - 1, -1, -1):
        if is_saturated[i]:
            in_contamination_zone = True
            contamination[i] = 1.0
        elif in_contamination_zone:
            if is_above_reject[i]:
                in_contamination_zone = False
            else:
                contamination[i] = 1.0

    # Convert contamination to weight (high contamination = low weight)
    w_contamination = 1.0 - contamination

    # Combined weight
    weights = w_depth * w_contamination

    return weights


def get_molecule_weights(transmission: np.ndarray,
                         molecule: str,
                         config_path: Optional[str] = None) -> np.ndarray:
    """
    Compute smooth weights for a specific molecule.

    Convenience wrapper that loads molecule parameters from config
    and calls get_transmission_weights.

    Parameters
    ----------
    transmission : np.ndarray
        1D array of transmission values (0 to 1)
    molecule : str
        Molecule name (e.g., 'H2O', 'CH4', 'CO2', 'O2')
    config_path : str, optional
        Path to telluric config YAML

    Returns
    -------
    weights : np.ndarray
        Float array with values 0 to 1
    """
    params = get_molecule_params(molecule, config_path)

    return get_transmission_weights(
        transmission,
        depth_max=params['depth_max'],
        depth_saturated=params['depth_saturated'],
        reject_saturated=params['reject_saturated']
    )


def load_telluric_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load telluric configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to telluric config YAML. Defaults to telluric_config.yaml
        in the same directory as this module.

    Returns
    -------
    config : dict
        Configuration dictionary with molecule parameters
    """
    if config_path is None:
        config_path = TELLURIC_CONFIG_PATH

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_molecule_params(molecule: str,
                        config_path: Optional[str] = None) -> Dict[str, float]:
    """
    Get absorption parameters for a specific molecule from config.

    Parameters
    ----------
    molecule : str
        Molecule name (e.g., 'H2O', 'CH4', 'CO2', 'O2')
    config_path : str, optional
        Path to telluric config YAML

    Returns
    -------
    params : dict
        Dictionary with depth_max, depth_saturated, reject_saturated
    """
    config = load_telluric_config(config_path)

    if molecule not in config.get('molecules', {}):
        tprint(f"Warning: Molecule '{molecule}' not in config, using defaults", color='orange')
        return {
            'depth_max': 0.5,
            'depth_saturated': 0.2,
            'reject_saturated': 0.8
        }

    return config['molecules'][molecule]


def get_valid_molecule_mask(transmission: np.ndarray,
                            molecule: str,
                            config_path: Optional[str] = None) -> np.ndarray:
    """
    Create a boolean mask of valid pixels for a specific molecule.

    Convenience wrapper that loads molecule parameters from config
    and calls get_valid_transmission_mask.

    Parameters
    ----------
    transmission : np.ndarray
        1D array of transmission values (0 to 1)
    molecule : str
        Molecule name (e.g., 'H2O', 'CH4', 'CO2', 'O2')
    config_path : str, optional
        Path to telluric config YAML

    Returns
    -------
    valid_mask : np.ndarray
        Boolean array where True = valid pixel, False = invalid pixel
    """
    params = get_molecule_params(molecule, config_path)

    return get_valid_transmission_mask(
        transmission,
        depth_max=params['depth_max'],
        depth_saturated=params['depth_saturated'],
        reject_saturated=params['reject_saturated']
    )


def get_header_transm(instrument: str = DEFAULT_INSTRUMENT) -> fits.Header:
    """
    Get TAPAS transmission file header.

    Parameters
    ----------
    instrument : str
        Instrument name

    Returns
    -------
    header : fits.Header
        TAPAS file header
    """
    params = get_user_params(instrument)
    calib_paths = get_calib_paths(instrument, params['project_path'])
    return fits.getheader(calib_paths['tapas_file'])


# ============================================================================
# FITS I/O Utilities
# ============================================================================

def getdata_safe(filename: str, ext: Optional[Union[int, str]] = None) -> np.ndarray:
    """
    Safely load FITS data with proper file closure.

    Opens a FITS file and returns a copy of the data to avoid issues
    with closed file handles.

    Parameters
    ----------
    filename : str
        Path to FITS file
    ext : int or str, optional
        Extension to read. If None, returns first extension with data.

    Returns
    -------
    data : np.ndarray
        Copy of FITS data

    Raises
    ------
    ValueError
        If file has no data or requested extension is empty
    FileNotFoundError
        If file does not exist
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"FITS file not found: {filename}")

    with fits.open(filename) as hdulist:
        if ext is None:
            # Return first HDU with data
            for hdu in hdulist:
                if hdu.data is not None:
                    return hdu.data.copy()
            raise ValueError(f"No data found in {filename}")
        else:
            data = hdulist[ext].data
            if data is None:
                raise ValueError(f"No data in extension {ext} of {filename}")
            return data.copy()


def getheader_safe(filename: str, ext: int = 0) -> fits.Header:
    """
    Safely load FITS header with proper file closure.

    Parameters
    ----------
    filename : str
        Path to FITS file
    ext : int
        Extension number (default 0 for primary)

    Returns
    -------
    header : fits.Header
        Copy of FITS header

    Raises
    ------
    FileNotFoundError
        If file does not exist
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"FITS file not found: {filename}")

    with fits.open(filename) as hdulist:
        return hdulist[ext].header.copy()


# ============================================================================
# Sky Emission Reconstruction (PCA Method - Optimized)
# ============================================================================

def sky_pca_fast(wave: Optional[np.ndarray] = None,
                spectrum: Optional[np.ndarray] = None,
                sky_dict: Optional[Dict] = None,
                force_positive: bool = True,
                doplot: bool = False,
                verbose: bool = True,
                instrument: str = DEFAULT_INSTRUMENT) -> Union[Dict, np.ndarray]:
    """
    Fast sky PCA reconstruction using analytical gradients.

    This is an optimized version with 10-100x faster convergence compared
    to the original numerical gradient approach.

    Parameters
    ----------
    wave : np.ndarray, optional
        Wavelength grid (n_orders, n_pixels)
    spectrum : np.ndarray, optional
        Spectrum to fit (same shape as wave)
    sky_dict : dict, optional
        Dictionary with 'SCI_SKY' and 'WAVE'. If None, loads from files
        and returns dictionary.
    force_positive : bool
        If True, force sky amplitudes to be non-negative
    doplot : bool
        If True, show diagnostic plot
    verbose : bool
        If True, print progress information
    instrument : str
        Instrument name

    Returns
    -------
    sky_out : np.ndarray or dict
        Fitted sky model (same shape as spectrum), or sky_dict if loading

    Notes
    -----
    The sky emission is dominated by OH airglow lines. This function
    uses PCA components to model the sky efficiently.

    Optimization strategy:
    - Analytical gradient for L-BFGS-B (10-100x faster)
    - Least squares initialization
    - Pre-flattened arrays to avoid repeated .ravel() calls
    - Robust weighting based on residual distribution
    """
    # If no sky_dict provided, load and return it
    if sky_dict is None:
        params = get_user_params(instrument)
        calib_paths = get_calib_paths(instrument, params['project_path'])

        sky_file = os.path.join(
            params['project_path'],
            f'sky_{instrument}/sky_pca_components.fits'
        )

        sky_dict = {
            'SCI_SKY': fits.getdata(sky_file),
            'WAVE': fits.getdata(calib_paths['waveref_file'])
        }
        return sky_dict

    # Get number of PCA components
    Npca = sky_dict['SCI_SKY'].shape[0]

    # Interpolate PCA components onto observation wavelength grid
    cube = np.zeros((Npca, *wave.shape))
    for ipca in range(Npca):
        cube[ipca] = wavecore.wave_to_wave(
            sky_dict['SCI_SKY'][ipca].reshape(wave.shape),
            sky_dict['WAVE'],
            wave
        )

    # Pre-flatten arrays (major optimization)
    wave_flat = wave.ravel()
    spectrum_flat = spectrum.ravel().astype(np.float64)
    cube_flat = cube.reshape(Npca, -1).astype(np.float64)

    sky_out = np.zeros_like(spectrum_flat)

    # Fit sky in spectral bands
    bands = SKY_PCA_CONFIG['bands']

    for wavemin, wavemax, band_name in bands:
        # Define spectral domain
        domain = (wave_flat > wavemin) & (wave_flat < wavemax)
        n_domain = np.sum(domain)

        # Extract domain data (contiguous for speed)
        spec_dom = np.ascontiguousarray(spectrum_flat[domain])
        cube_dom = np.ascontiguousarray(cube_flat[:, domain])

        # Valid pixel mask (no NaN)
        valid_mask = np.isfinite(spec_dom) & np.all(np.isfinite(cube_dom), axis=0)
        n_valid = np.sum(valid_mask)

        # NaN-safe cube for gradient (precomputed once)
        cube_dom_safe = np.where(np.isfinite(cube_dom), cube_dom, 0)

        if n_valid < Npca:
            # Not enough valid pixels in this domain, skip
            continue

        # Least squares initialization (better than zero guess)
        spec_valid = spec_dom[valid_mask]
        cube_valid = cube_dom[:, valid_mask]
        x0 = np.linalg.lstsq(cube_valid.T, spec_valid, rcond=None)[0]

        def compute_sky(amps: np.ndarray) -> np.ndarray:
            """Compute sky model from PCA amplitudes."""
            sky = np.dot(amps, cube_dom_safe)
            if force_positive:
                sky = np.maximum(sky, 0)
            return sky

        def objective_and_gradient(amps: np.ndarray) -> Tuple[float, np.ndarray]:
            """
            Compute objective function and analytical gradient.

            This is the key optimization: analytical gradient is ~10x faster
            than numerical approximation.
            """
            sky = compute_sky(amps)
            residual = spec_dom - sky

            # Robust RMS estimation via MAD of differences
            res_valid = residual[valid_mask]
            rms = np.nanmedian(np.abs(np.diff(res_valid))) + 1e-10

            # Robust weights (down-weight outliers)
            nsig = res_valid / rms
            p_valid_prob = np.exp(-0.5 * nsig**2)
            weights = p_valid_prob / (p_valid_prob + PROCESSING_DEFAULTS['robust_weight_threshold'])

            # Weighted chi-square objective
            obj = np.sum(weights * res_valid**2) / n_valid

            # Analytical gradient
            weighted_res = np.zeros(n_domain)
            weighted_res[valid_mask] = weights * res_valid

            # Account for positivity constraint
            if force_positive:
                sky_unclipped = np.dot(amps, cube_dom_safe)
                weighted_res = weighted_res * (sky_unclipped >= 0)

            grad = -2.0 * np.dot(cube_dom_safe, weighted_res) / n_valid

            return obj, grad

        # Optimize using L-BFGS-B with analytical gradient
        result = minimize(
            objective_and_gradient,
            x0,
            method='L-BFGS-B',
            jac=True,  # Use analytical gradient
            options={
                'maxiter': SKY_PCA_CONFIG['max_iterations'],
                'ftol': SKY_PCA_CONFIG['ftol'],
                'gtol': SKY_PCA_CONFIG['gtol']
            }
        )

        # Apply final model
        sky_out[domain] += compute_sky(result.x)

    # Diagnostic plot
    if doplot:
        plt.figure(figsize=(12, 4))
        plt.plot(wave_flat, spectrum_flat - sky_out, 'g', alpha=0.8, lw=1,
                label='Sky residual')
        plt.plot(wave_flat, spectrum_flat, 'b', alpha=0.5, lw=0.5,
                label='Spectrum')
        plt.plot(wave_flat, sky_out, 'r', alpha=0.8, lw=0.5, label='Sky model')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux')
        plt.legend()
        plt.title(f'Sky PCA Fit')
        plt.tight_layout()
        plt.show()

    return sky_out.reshape(spectrum.shape)


def sky_pca(wave: Optional[np.ndarray] = None,
           spectrum: Optional[np.ndarray] = None,
           sky_dict: Optional[Dict] = None,
           force_positive: bool = True,
           doplot: bool = False,
           instrument: str = DEFAULT_INSTRUMENT) -> Union[Dict, np.ndarray]:
    """
    Original sky PCA method (kept for compatibility).

    This is the original implementation. Use sky_pca_fast() for better
    performance (10-100x faster).

    Parameters
    ----------
    wave : np.ndarray, optional
        Wavelength grid
    spectrum : np.ndarray, optional
        Spectrum to fit
    sky_dict : dict, optional
        Sky PCA components dictionary
    force_positive : bool
        Force positive sky amplitudes
    doplot : bool
        Show diagnostic plot
    instrument : str
        Instrument name

    Returns
    -------
    sky_out : np.ndarray or dict
        Sky model or dictionary
    """
    # Load sky dictionary if not provided
    if sky_dict is None:
        params = get_user_params(instrument)
        sky_file = os.path.join(
            params['project_path'],
            f'sky_{instrument}/sky_pca_components.fits'
        )
        calib_paths = get_calib_paths(instrument, params['project_path'])

        sky_dict = {
            'SCI_SKY': fits.getdata(sky_file),
            'WAVE': fits.getdata(calib_paths['waveref_file'])
        }
        return sky_dict

    Npca = sky_dict['SCI_SKY'].shape[0]
    cube = np.zeros((Npca, *wave.shape))

    for ipca in range(Npca):
        cube[ipca] = wavecore.wave_to_wave(
            sky_dict['SCI_SKY'][ipca].reshape(wave.shape),
            sky_dict['WAVE'],
            wave
        )

    sky_out = np.zeros_like(spectrum).ravel()

    for Y_JH in range(2):
        # Define band
        if Y_JH == 0:
            wavemin, wavemax = 950, 1400
            # Silently fitting sky in Y+J band
        else:
            wavemin, wavemax = 1400, 1900
            # Silently fitting sky in H band

        domain = (wave.ravel() > wavemin) & (wave.ravel() < wavemax)

        # Initialize amplitudes
        x0 = np.zeros(Npca)
        for iamp in range(Npca):
            x0[iamp] = (np.nansum(spectrum.ravel()[domain] * cube[iamp].ravel()[domain]) /
                       np.nansum(cube[iamp].ravel()[domain]**2))

        def apply_amps(amps: np.ndarray) -> np.ndarray:
            """Apply PCA amplitudes."""
            sky0 = np.zeros(cube[0].shape).ravel()
            for ipca in range(Npca):
                sky0[domain] += cube[ipca].ravel()[domain] * amps[ipca]

            if force_positive:
                sky0[domain][sky0[domain] < 0] = 0.0
            return sky0

        # Silently optimizing sky PCA amplitudes in this band

        def model_q(amps: np.ndarray) -> float:
            """Objective function."""
            diff = (spectrum.ravel() - apply_amps(amps))[domain]
            rms = np.nanmedian(np.abs(np.diff(diff)))
            nsig = diff / rms
            p_valid = np.exp(-0.5 * nsig**2)
            p_invalid = 1e-4
            w = p_valid / (p_valid + p_invalid)

            d2 = np.nanmean(w * (diff)**2)
            return d2

        # Optimize
        x = minimize(model_q, x0)
        sky_out += apply_amps(x.x)

    sky_out = sky_out.reshape(spectrum.shape)

    if doplot:
        plt.plot(wave.ravel(), (spectrum - sky_out).ravel(),
                label='Original Spectrum', color='blue')
        plt.plot(wave.ravel(), sky_out.ravel(),
                label='Reconstructed Sky', color='red', alpha=0.5)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux')
        plt.legend()
        plt.title('Sky PCA Reconstruction (Original Method)')
        plt.show()

    return sky_out


# ============================================================================
# Velocity Determination
# ============================================================================

def gauss(x: np.ndarray, a: float, x0: float, sigma: float,
         zp: float, expo: float) -> np.ndarray:
    """
    Super-Gaussian function for velocity CCF fitting.

    Parameters
    ----------
    x : np.ndarray
        Input values (velocities)
    a : float
        Amplitude
    x0 : float
        Center position
    sigma : float
        Width parameter
    zp : float
        Zero-point offset
    expo : float
        Exponent (2 = Gaussian, >2 = super-Gaussian)

    Returns
    -------
    y : np.ndarray
        Function values
    """
    return a * np.exp(-0.5 * np.abs((x - x0) / sigma)**expo) + zp


def get_velo(wave: np.ndarray,
            sp: np.ndarray,
            spl: ius,
            dv_amp: float = VELOCITY_CONFIG['dv_amp_default'],
            doplot: bool = True) -> float:
    """
    Determine stellar velocity by cross-correlation with template.

    Uses a two-stage approach:
    1. Coarse search every 10 steps
    2. Fine search around peak

    Parameters
    ----------
    wave : np.ndarray
        Wavelength grid (n_orders, n_pixels)
    sp : np.ndarray
        Observed spectrum (same shape as wave)
    spl : InterpolatedUnivariateSpline
        Template spline interpolator
    dv_amp : float
        Velocity search range (±dv_amp in km/s)
    doplot : bool
        If True, show diagnostic plot

    Returns
    -------
    velo : float
        Optimal velocity shift (km/s)

    Notes
    -----
    The velocity is determined by maximizing the correlation between
    the high-pass filtered (log) spectrum and template.
    """
    # Create velocity grid
    dvs = np.arange(-dv_amp, dv_amp + 1, VELOCITY_CONFIG['dv_step'], dtype=float)

    # High-pass filter spectrum in log space
    with np.errstate(invalid='ignore'):
        sp_tmp = np.log(sp).ravel()
    sp_tmp -= mp.lowpassfilter(sp_tmp, 101)

    amp = np.zeros_like(dvs, dtype=float) + np.nan
    rms = mp.robust_nanstd(np.diff(sp_tmp))

    # Coarse search (every 10 steps)
    for i in tqdm(range(len(dvs))[::VELOCITY_CONFIG['coarse_step']],
                 desc='Optimizing velocity shift (coarse)', leave=False):
        dv = dvs[i]
        template2 = np.log(spl(wave * mp.relativistic_waveshift(dv))).ravel()
        amp[i] = np.nansum(sp_tmp * template2)

    # Find coarse peak
    v0 = dvs[np.nanargmax(amp)]

    # Fine search around peak
    for i in tqdm(range(len(dvs)), desc='Optimizing velocity shift (fine)', leave=False):
        if np.isfinite(amp[i]):
            continue
        if np.abs(dvs[i] - v0) > VELOCITY_CONFIG['fine_range']:
            continue

        dv = dvs[i]
        template2 = np.log(spl(wave * mp.relativistic_waveshift(dv))).ravel()
        amp[i] = np.nansum(sp_tmp * template2)

    # Remove NaN values
    keep = np.isfinite(amp)
    dvs = dvs[keep]
    amp = amp[keep]

    # Fit super-Gaussian to CCF peak
    p0 = [np.nanmax(amp), dvs[np.nanargmax(amp)], 5.0, 0, 2]
    popt, pcov = curve_fit(gauss, dvs, amp, p0=p0)

    tprint(f'  Optimal velocity shift: {popt[1]:.2f} km/s', color='blue')

    # Diagnostic plot
    if doplot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dvs, amp, 'o', label='Cross-correlation', alpha=0.7)
        ax.plot(dvs, gauss(dvs, *popt), 'r-', label='Super-Gaussian fit', lw=2)
        ax.axvline(popt[1], color='k', linestyle='--', alpha=0.5, label=f'Peak: {popt[1]:.2f} km/s')
        ax.set_xlabel('Velocity shift (km/s)')
        ax.set_ylabel('Cross-correlation amplitude')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title('Stellar Velocity Determination')
        plt.tight_layout()
        plt.show()

    return popt[1]


# ============================================================================
# Header Management
# ============================================================================

def update_header(hdr: fits.Header, instrument: str = 'NIRPS') -> fits.Header:
    """
    Update FITS header with computed atmospheric and observational parameters.

    Adds or updates the following keywords:
    - AIRMASS: Mean airmass during observation
    - ACCAIRM: Accurate airmass accounting for curvature
    - PRESSURE: Ambient pressure
    - TEMPERAT: Ambient temperature (Kelvin)
    - SUNSETD: Delay since last sunset (hours)
    - HUMIDITY: Relative humidity
    - NORMPRES: Normalization pressure for TAPAS values
    - SNR_REF: Reference SNR
    - H2OCV, VMR_CO2, VMR_CH4: TAPAS molecular values

    Parameters
    ----------
    hdr : fits.Header
        Input FITS header
    instrument : str
        Instrument name ('NIRPS' or 'SPIROU')

    Returns
    -------
    hdr : fits.Header
        Updated header

    Notes
    -----
    This function standardizes header keywords across instruments and
    adds computed quantities needed for telluric modeling.
    """
    keys = HEADER_KEYS[instrument]

    # Airmass
    if instrument == 'NIRPS':
        airmass = (hdr[f'HIERARCH {keys["airmass_start"]}'] +
                  hdr[f'HIERARCH {keys["airmass_end"]}']) / 2.0
        hdr['AIRMASS'] = (airmass, 'Mean airmass during observation')
    elif instrument == 'SPIROU':
        airmass = hdr[keys['airmass']]
        # Already present, no action needed
    else:
        raise ValueError(f'Unknown instrument: {instrument}')

    # Accurate airmass (accounting for atmospheric curvature)
    hdr['ACCAIRM'] = (accurate_airmass(hdr), 'Accurate airmass from MJDMID')

    # Pressure
    if instrument == 'NIRPS':
        pressure = (hdr[f'HIERARCH {keys["pressure_start"]}'] +
                   hdr[f'HIERARCH {keys["pressure_end"]}']) / 2.0
        hdr['PRESSURE'] = (pressure, '[kPa] Ambient pressure at telescope')
    elif instrument == 'SPIROU':
        pressure = hdr[keys['pressure']]

    # Temperature (convert to Kelvin)
    if instrument == 'NIRPS':
        temp_celsius = hdr[f'HIERARCH {keys["temperature"]}']
        hdr['TEMPERAT'] = (temp_celsius + 273.15, 'Ambient temperature [K]')
    elif instrument == 'SPIROU':
        temp_celsius = hdr[keys['temperature']]
        hdr['TEMPERAT'] = (temp_celsius + 273.15, 'Ambient temperature [K]')

    # Delay since sunset
    hdr['SUNSETD'] = (delay_since_sunset(hdr['MJDMID'], hdr=hdr),
                     'Delay since last sunset [hours]')

    # Humidity
    if instrument == 'NIRPS':
        hdr['HUMIDITY'] = (hdr[f'HIERARCH {keys["humidity"]}'],
                          'Relative humidity [%]')
    elif instrument == 'SPIROU':
        hdr['HUMIDITY'] = hdr[keys['humidity']]

    # TAPAS reference values
    params = get_user_params(instrument)
    calib_paths = get_calib_paths(instrument, params['project_path'])
    tapas_file = calib_paths['tapas_file']
    hdr_transm = fits.getheader(tapas_file)

    pressure0 = hdr_transm['PAMBIENT']
    hdr['NORMPRES'] = (pressure0, '[kPa] Normalization pressure for TAPAS values')

    # SNR reference
    hdr['SNR_REF'] = (hdr[keys['snr_ref']],
                     'SNR on reference order at ~1.60 micron')

    # TAPAS molecular values
    hdr['H2OCV'] = (hdr_transm['H2OCV'], 'TAPAS H2O column value [cm]')
    hdr['VMR_CO2'] = (hdr_transm['VMR_CO2'], 'TAPAS CO2 volume mixing ratio [ppm]')
    hdr['VMR_CH4'] = (hdr_transm['VMR_CH4'], 'TAPAS CH4 volume mixing ratio [ppm]')

    return hdr


def hotstar(hdr: fits.Header) -> fits.Header:
    """
    Identify if object is a hot star from predefined list.

    Hot stars have negligible absorption lines, making them ideal
    for telluric calibration.

    Parameters
    ----------
    hdr : fits.Header
        FITS header

    Returns
    -------
    hdr : fits.Header
        Header with 'HOTSTAR' keyword added
    """
    hot_star_list = TEMPLATE_CONFIG['hot_star_list']

    drsobj = hdr['DRSOBJN'].strip()

    hdr['HOTSTAR'] = drsobj in hot_star_list

    return hdr


# ============================================================================
# Airmass and Sunset Calculations
# ============================================================================

def accurate_airmass(hdr: fits.Header) -> float:
    """
    Calculate accurate airmass accounting for atmospheric curvature.

    Uses the formula from Kasten & Young (1989) which accounts for
    Earth's curvature and atmospheric refraction.

    Parameters
    ----------
    hdr : fits.Header
        FITS header with observation coordinates and time

    Returns
    -------
    airmass : float
        Accurate airmass value

    Notes
    -----
    The simple sec(z) formula breaks down at high airmass.
    This function uses a more accurate model valid up to ~80° zenith angle.
    """
    # Observatory location
    lat = hdr['BC_LAT']
    lon = hdr['BC_LONG']
    height = hdr['BC_ALT']  # meters

    site = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)

    # Observation time
    mjd_obs = hdr['MJDMID']
    time = Time(mjd_obs, format='mjd')

    # AltAz frame at observation time
    altaz_frame = AltAz(obstime=time, location=site)

    # Object coordinates
    ra = hdr['PP_RA']
    dec = hdr['PP_DEC']
    obj_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    obj_altaz = obj_coord.transform_to(altaz_frame)

    # Zenith angle
    z = 90.0 - obj_altaz.alt.degree

    # Accurate airmass formula (accounts for curvature)
    R_earth = AIRMASS_CONFIG['R_earth']  # km
    H_atmo = AIRMASS_CONFIG['H_atmo']    # km
    z_rad = np.radians(z)
    sec_z = 1.0 / np.cos(z_rad)

    am = np.sqrt(
        (R_earth / (R_earth + H_atmo))**2 * sec_z**2 -
        (R_earth / (R_earth + H_atmo))**2 + 1.0
    )

    return am


def delay_since_sunset(mjd_obs: float, hdr: fits.Header) -> float:
    """
    Calculate time elapsed since last sunset at observatory.

    This is useful for tracking atmospheric conditions that change
    after sunset (e.g., water vapor settling).

    Parameters
    ----------
    mjd_obs : float
        MJD of observation
    hdr : fits.Header
        Header with observatory location

    Returns
    -------
    dt_hours : float
        Hours since sunset (positive after sunset)
    """
    # Observatory location
    lat = hdr['BC_LAT']
    lon = hdr['BC_LONG']
    height = hdr['BC_ALT']
    site = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)

    # Observation time
    time = Time(mjd_obs, format='mjd')

    # Check 24 hours around observation
    delta_hours = np.linspace(0, 24, 1000)
    times = time + delta_hours * u.hour

    altaz_frame = AltAz(obstime=times, location=site)
    sun_altaz = get_sun(times).transform_to(altaz_frame)

    # Find sunset (altitude crosses 0 from positive to negative)
    altitudes = sun_altaz.alt.degree
    sunset_idx = np.where((altitudes[:-1] > 0) & (altitudes[1:] < 0))[0]

    if len(sunset_idx) > 0:
        sunset_time = times[sunset_idx[0]]
    else:
        # No sunset found, return NaN
        return np.nan

    # Time since sunset
    dt = mjd_obs - sunset_time.mjd
    dt_hours = dt * 24.0

    # If negative (before sunset), add 24h to get previous sunset
    if dt_hours < 0:
        dt_hours += 24.0

    return dt_hours


# ============================================================================
# Weight and Smoothing Functions
# ============================================================================

def weight_fall(x: np.ndarray, knee: float, w: Optional[float] = None) -> np.ndarray:
    """
    Smooth sigmoid transition function for absorption weighting.

    Creates a smooth transition from 0 → 0.5 → 1 centered at 'knee'.
    Used to down-weight regions with strong absorption.

    Parameters
    ----------
    x : np.ndarray
        Input values (typically transmission)
    knee : float
        Midpoint of transition (where weight = 0.5)
    w : float, optional
        Transition width. If None, uses knee/5.

    Returns
    -------
    weights : np.ndarray
        Weight values between 0 and 1
    """
    if w is None:
        w = knee / 5.0

    return ne.evaluate('1 / (1 + exp(-4 * (x - knee) / w))')


def weighted_nanstd(data: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted standard deviation, ignoring NaN values.
    
    Uses numexpr for speed on large arrays.
    
    Parameters
    ----------
    data : np.ndarray
        Data values (can contain NaN)
    weights : np.ndarray
        Weight values (can contain NaN, must be same shape as data)
        
    Returns
    -------
    std : float
        Weighted standard deviation
        
    Notes
    -----
    Uses the reliability weights formula with Bessel-like correction:
        variance = sum(w * (x - mean)^2) / (sum(w) - sum(w^2)/sum(w))
    """
    # Flatten for speed
    data = np.asarray(data).ravel()
    weights = np.asarray(weights).ravel()
    
    # Valid mask (both data and weights finite)
    valid = np.isfinite(data) & np.isfinite(weights) & (weights > 0)
    
    if np.sum(valid) < 2:
        return np.nan
    
    d = data[valid]
    w = weights[valid]
    
    # Use numexpr for speed
    w_sum = ne.evaluate('sum(w)')
    w_sum2 = ne.evaluate('sum(w * w)')
    
    # Weighted mean
    mean = ne.evaluate('sum(w * d)') / w_sum
    
    # Weighted variance with Bessel correction
    diff = d - mean
    variance = ne.evaluate('sum(w * diff * diff)') / (w_sum - w_sum2 / w_sum)
    
    return np.sqrt(variance)


def weighted_nanmean(data: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted mean, ignoring NaN values.
    
    Uses numexpr for speed on large arrays.
    
    Parameters
    ----------
    data : np.ndarray
        Data values (can contain NaN)
    weights : np.ndarray
        Weight values (can contain NaN, must be same shape as data)
        
    Returns
    -------
    mean : float
        Weighted mean
    """
    # Flatten for speed
    data = np.asarray(data).ravel()
    weights = np.asarray(weights).ravel()
    
    # Valid mask
    valid = np.isfinite(data) & np.isfinite(weights) & (weights > 0)
    
    if np.sum(valid) == 0:
        return np.nan
    
    d = data[valid]
    w = weights[valid]
    
    return ne.evaluate('sum(w * d)') / ne.evaluate('sum(w)')


def savgol_filter_nan_fast(y: np.ndarray,
                           window_length: int,
                           polyorder: int,
                           deriv: int = 0,
                           frac_valid: float = 0.3) -> np.ndarray:
    """
    Savitzky-Golay filter that handles NaN values.

    Fits a polynomial to each window using only valid (non-NaN) samples.

    Parameters
    ----------
    y : np.ndarray
        Data to filter (NaNs allowed)
    window_length : int
        Filter window length (must be odd)
    polyorder : int
        Polynomial order
    deriv : int
        Derivative order (0 = smoothed value)
    frac_valid : float
        Minimum fraction of valid samples required

    Returns
    -------
    y_filtered : np.ndarray
        Filtered data (NaN where insufficient valid samples)
    """
    y = np.asarray(y)
    half_window = window_length // 2
    y_filtered = np.full_like(y, np.nan, dtype=np.float64)
    y_padded = np.pad(y, half_window, mode='edge')

    for i in range(len(y)):
        window_data = y_padded[i:i + window_length]
        valid_mask = ~np.isnan(window_data)
        n_valid = np.sum(valid_mask)
        fraction_valid = n_valid / window_length

        if fraction_valid < frac_valid or n_valid <= polyorder:
            continue

        # Fit polynomial using valid points
        x_valid = np.arange(window_length)[valid_mask] - half_window
        A = np.vander(x_valid, polyorder + 1, increasing=True)
        y_valid = window_data[valid_mask]

        coeffs, _, _, _ = np.linalg.lstsq(A, y_valid, rcond=None)

        # Evaluate at center
        if deriv == 0:
            y_filtered[i] = coeffs[0]
        else:
            if deriv <= polyorder:
                factorial = np.math.factorial(deriv)
                y_filtered[i] = coeffs[deriv] * factorial
            else:
                y_filtered[i] = 0

    return y_filtered


# ============================================================================
# Template Management
# ============================================================================

def fetch_template(hdr: fits.Header,
                  wavemin: Optional[float] = None,
                  wavemax: Optional[float] = None,
                  instrument: str = DEFAULT_INSTRUMENT) -> Tuple[ius, ius]:
    """
    Fetch stellar template spectrum based on effective temperature.

    Interpolates between model spectra at bracketing temperatures.

    Parameters
    ----------
    hdr : fits.Header
        FITS header with PP_TEFF (effective temperature)
    wavemin : float, optional
        Minimum wavelength (nm)
    wavemax : float, optional
        Maximum wavelength (nm)
    instrument : str
        Instrument name (for wavelength range defaults)

    Returns
    -------
    spline : InterpolatedUnivariateSpline
        Template flux spline
    spline_dv : InterpolatedUnivariateSpline
        Template velocity gradient spline

    Notes
    -----
    Templates are from a grid of synthetic spectra (3000-6000 K in 500 K steps).
    Linear interpolation is performed between bracketing temperatures.
    """
    # Extract stellar temperature
    teff = hdr['PP_TEFF']

    # Fallback for RV if pipeline value is zero
    if hdr['PP_RV'] == 0 and hdr.get('HIERARCH ESO TEL TARG RADVEL', 0) != 0:
        tprint("Using HIERARCH ESO TEL TARG RADVEL for systemic velocity", color='blue')
        hdr['PP_RV'] = hdr['HIERARCH ESO TEL TARG RADVEL']

    # Set wavelength range
    if instrument.upper() == 'NIRPS':
        wave1 = 950 if wavemin is None else wavemin
        wave2 = 2000 if wavemax is None else wavemax
    elif instrument.upper() == 'SPIROU':
        wave1 = 950 if wavemin is None else wavemin
        wave2 = 2550 if wavemax is None else wavemax
    else:
        raise ValueError(f"Unknown instrument {instrument}")

    # Round to temperature grid
    temp_config = TEMPLATE_CONFIG
    t_up = int(teff / temp_config['temp_step'] + 1) * temp_config['temp_step']
    t_low = t_up - temp_config['temp_step']

    # Enforce limits
    t_low = max(t_low, temp_config['temp_min'])
    t_up = min(t_up, temp_config['temp_max'])

    # Load models
    params = get_user_params(instrument)
    filename_low = os.path.join(params['project_path'],
                               f'models/temperature_gradient_{t_low}.fits')
    filename_up = os.path.join(params['project_path'],
                              f'models/temperature_gradient_{t_up}.fits')

    # Interpolation weights
    weight_up = (teff - t_low) / (t_up - t_low) if t_up != t_low else 0.0
    weight_low = 1.0 - weight_up

    # Read tables
    tbl_low = Table.read(filename_low)
    tbl_up = Table.read(filename_up)

    # Extract and interpolate
    wave = np.array(tbl_low['wavelength'])
    log_flux_low = np.array(tbl_low['flux'])
    log_flux_up = np.array(tbl_up['flux'])

    log_flux = log_flux_low * weight_low + log_flux_up * weight_up

    # Velocity gradient
    dv = np.gradient(log_flux) / np.gradient(wave) * SPEED_OF_LIGHT

    # Filter wavelength range
    keep = ((wave >= wave1) & (wave <= wave2) &
            np.isfinite(log_flux) & np.isfinite(dv))

    wave = wave[keep]
    flux = np.exp(log_flux[keep])
    dv = dv[keep]

    # Create splines
    spline = ius(wave, flux, k=1, ext=3)
    spline_dv = ius(wave, dv, k=1, ext=3)

    return spline, spline_dv


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
    Convolve spectrum with variable resolution kernel (vectorized).

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

    n_pix = len(spectrum)

    # Determine scan range
    scale1 = np.max(res_fwhm)
    scale2 = np.median(np.gradient(wavemap) / wavemap) * SPEED_OF_LIGHT
    range_scan = CONVOLUTION_CONFIG['range_scan_scale'] * (scale1 / scale2)
    range_scan = int(np.ceil(range_scan))

    # Mask NaN pixels
    valid_pix = np.isfinite(spectrum)
    spectrum_clean = np.where(valid_pix, spectrum, 0.0)
    valid_pix_float = valid_pix.astype(float)

    # Super-Gaussian width parameter
    ew = (res_fwhm / 2) / (2 * np.log(2))**(1 / res_expo)

    # Build all offsets at once (sorted by distance from center)
    offsets = np.arange(-range_scan, range_scan)
    offsets = offsets[np.argsort(np.abs(offsets))]

    # Pre-compute all rolled wavelengths and spectra
    # Shape: (n_offsets, n_pix)
    rolled_wave = np.array([np.roll(wavemap, off) for off in offsets])
    rolled_spec = np.array([np.roll(spectrum_clean, off) for off in offsets])
    rolled_valid = np.array([np.roll(valid_pix_float, off) for off in offsets])

    # Compute velocity shifts for all offsets: (n_offsets, n_pix)
    dv = SPEED_OF_LIGHT * (wavemap[np.newaxis, :] / rolled_wave - 1)

    # Compute kernels for all offsets: (n_offsets, n_pix)
    # Using numexpr for speed
    ew_broad = ew[np.newaxis, :]  # (1, n_pix)
    expo_broad = res_expo[np.newaxis, :]  # (1, n_pix)
    kernels = ne.evaluate('exp(-0.5 * abs(dv / ew_broad) ** expo_broad)')

    # Apply valid pixel mask to kernels
    kernels *= rolled_valid

    # Find where kernels are still significant (use max across pixels)
    kernel_max = np.max(kernels, axis=1)
    significant = kernel_max >= ker_thres

    # Only use significant offsets
    kernels = kernels[significant]
    rolled_spec = rolled_spec[significant]

    # Vectorized convolution sum
    spectrum2 = np.sum(rolled_spec * kernels, axis=0)
    sumker = np.sum(kernels, axis=0)

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

# Cache for convolved molecule transmissions
# Key: (molecule_index, exponent rounded to 3 decimals)
# Value: convolved transmission array
_conv_cache = {}
_conv_cache_max_size = 50  # Limit cache size to avoid memory issues

def clear_conv_cache():
    """Clear the convolution cache."""
    global _conv_cache
    _conv_cache = {}


# ============================================================================
# Precomputed Absorption Grid System
# ============================================================================

# Exponent grids for precomputation (log-uniform sampling)
# Non-water: 0.9 * 1.1^N for N=0..13 (14 values, 0.9 to ~3.1)
# Water: 0.2 * 1.1^N for N=0..49 (50 values, 0.2 to ~21.4)
EXPO_GRID_CONFIG = {
    'H2O': 0.2 * (1.1 ** np.arange(50)),     # Water: 0.2 to ~21, 50 templates
    'CH4': 0.9 * (1.1 ** np.arange(14)),     # Methane: 0.9 to ~3.1, 14 templates
    'CO2': 0.9 * (1.1 ** np.arange(14)),     # CO2: 0.9 to ~3.1, 14 templates
    'O2': 0.9 * (1.1 ** np.arange(14)),      # O2: 0.9 to ~3.1, 14 templates
}

# Global storage for precomputed grid
_precomputed_grid = None
_precomputed_waveref = None


def get_precompute_path(instrument: str) -> str:
    """Get path to precomputed absorption grid file."""
    params = get_user_params(instrument)
    project_path = params['project_path']
    tmp_dir = os.path.join(project_path, f'tmp_{instrument}')
    os.makedirs(tmp_dir, exist_ok=True)
    return os.path.join(tmp_dir, 'precomputed_absorption_grid.pkl')


def precompute_absorption_grid(instrument: str = DEFAULT_INSTRUMENT, 
                               force_recompute: bool = False) -> Dict:
    """
    Precompute convolved absorption templates at discrete exponent values.
    
    This is run once at startup and saves results to disk. Subsequent runs
    load from disk instead of recomputing.
    
    Parameters
    ----------
    instrument : str
        Instrument name ('NIRPS' or 'SPIROU')
    force_recompute : bool
        If True, recompute even if cached file exists
        
    Returns
    -------
    grid : dict
        Dictionary with precomputed absorption arrays and weights
    """
    import pickle
    import time
    
    global _precomputed_grid, _precomputed_waveref
    
    pkl_path = get_precompute_path(instrument)
    
    # Get current molecule params from YAML for consistency check
    current_mol_params = {}
    for mol in MOLECULES:
        mp = get_molecule_params(mol)
        current_mol_params[mol] = {
            'depth_max': mp['depth_max'],
            'depth_saturated': mp['depth_saturated'],
            'reject_saturated': mp['reject_saturated']
        }
    
    # Try to load from disk
    if os.path.exists(pkl_path) and not force_recompute:
        tprint(f"Loading precomputed absorption grid from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if molecule params have changed (requires recomputation of weights)
        stored_mol_params = data.get('molecule_params', None)
        if stored_mol_params is None:
            tprint("  \033[33mWarning: Old pickle format without molecule params. Recomputing...\033[0m")
        elif stored_mol_params != current_mol_params:
            tprint("  \033[33mWarning: YAML molecule params changed. Recomputing grid...\033[0m")
            for mol in MOLECULES:
                if stored_mol_params.get(mol) != current_mol_params.get(mol):
                    tprint(f"    {mol}: stored={stored_mol_params.get(mol)} vs current={current_mol_params.get(mol)}")
        else:
            # Params match, use cached data
            _precomputed_grid = data['grid']
            _precomputed_waveref = data['waveref']
            n_templates = sum(len(v) for v in _precomputed_grid.values())
            tprint(f"  Loaded {n_templates} precomputed templates (params match YAML)")
            return _precomputed_grid
    
    # Need to compute
    tprint("="*60)
    tprint("PRECOMPUTING ABSORPTION GRID")
    tprint("  (This is done once per instrument, saved to disk)")
    tprint(f"  Output: {pkl_path}")
    tprint("="*60)
    
    t_start = time.time()
    params = get_user_params(instrument)
    project_path = params['project_path']
    
    # Load reference wavelength and TAPAS data
    tprint(f"Loading reference wavelength grid...")
    waveref = fits.getdata(os.path.join(project_path, f'calib_{instrument}/waveref.fits'))
    _precomputed_waveref = waveref
    tprint(f"  Shape: {waveref.shape} ({waveref.shape[0]} orders x {waveref.shape[1]} pixels)")
    
    # Load TAPAS absorption data
    if instrument == 'NIRPS':
        transm_file = 'LaSilla_NIRPS_tapas.fits'
    elif instrument == 'SPIROU':
        transm_file = 'MaunaKea_tapas.fits'
    else:
        raise ValueError(f'Unknown instrument: {instrument}')
    
    tprint(f"Loading TAPAS transmission: {transm_file}")
    transm_file = os.path.join(project_path, transm_file)
    transm_table = Table.read(transm_file)
    
    # Create table with relevant molecules
    tbl = Table()
    tbl['wavelength'] = transm_table['wavelength']
    for mol in MOLECULES:
        tbl[mol] = transm_table[mol]
    
    # Filter wavelength range
    keep_wave = ((tbl['wavelength'] >= np.min(waveref)) &
                (tbl['wavelength'] <= np.max(waveref)))
    tbl = tbl[keep_wave]
    
    transm_wave = np.array(tbl['wavelength'])
    
    # Interpolate base absorption onto reference grid
    tprint("Interpolating base absorption onto reference grid...")
    base_abso = {}
    for mol in MOLECULES:
        base_abso[mol] = ius(transm_wave, tbl[mol], ext=0, k=1)(waveref)
        base_abso[mol][base_abso[mol] < 0] = 0.0
    
    # Precompute grid
    grid = {}
    total_templates = sum(len(EXPO_GRID_CONFIG[mol]) for mol in MOLECULES)
    template_count = 0
    
    tprint(f"Computing {total_templates} templates for {len(MOLECULES)} molecules...")
    for mol in MOLECULES:
        mol_start = time.time()
        grid[mol] = {}
        expo_values = EXPO_GRID_CONFIG[mol]
        mol_params = get_molecule_params(mol)
        n_expo = len(expo_values)
        
        tprint(f"  {mol}: {n_expo} exponents ({expo_values[0]:.2f} to {expo_values[-1]:.2f})")
        tprint(f"    depth_max={mol_params['depth_max']}, depth_saturated={mol_params['depth_saturated']}, reject_saturated={mol_params['reject_saturated']}")
        
        for i_expo, expo in enumerate(expo_values):
            template_count += 1
            
            # Show progress on same line
            print(f"\r    Computing {mol} exponent {i_expo+1}/{n_expo}: {expo:.2f}", end='', flush=True)
            
            # Compute transmission with exponent
            mol_trans = base_abso[mol] ** expo
            
            # Convolve with instrumental resolution
            mol_trans_conv = variable_res_conv(waveref, mol_trans, E2DS_FWHM, E2DS_EXPO)
            
            # Compute weights
            weights = np.ones(waveref.shape, dtype=np.float64)
            for iord in range(waveref.shape[0]):
                order_weights = get_transmission_weights(
                    mol_trans_conv[iord],
                    depth_max=mol_params['depth_max'],
                    depth_saturated=mol_params['depth_saturated'],
                    reject_saturated=mol_params['reject_saturated']
                )
                weights[iord] = order_weights
            
            # Store with rounded key
            expo_key = round(expo, 3)
            grid[mol][expo_key] = {
                'transmission': mol_trans_conv,
                'weights': weights
            }
        
        # Clear progress line and show completion
        print("\r" + " "*60 + "\r", end='')
        mol_elapsed = time.time() - mol_start
        tprint(f"    Done: {n_expo} templates in {mol_elapsed:.1f}s")
    
    # Save to disk with molecule params for consistency check
    tprint(f"Saving precomputed grid to {pkl_path}...")
    data = {
        'grid': grid,
        'waveref': waveref,
        'instrument': instrument,
        'molecule_params': current_mol_params,  # For consistency check on reload
        'expo_grid_config': {mol: list(EXPO_GRID_CONFIG[mol]) for mol in MOLECULES}
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    
    _precomputed_grid = grid
    total_elapsed = time.time() - t_start
    tprint("="*60)
    tprint(f"Precomputation complete: {template_count} templates in {total_elapsed:.1f}s")
    tprint(f"Saved to: {pkl_path}")
    tprint("="*60)
    
    return grid


def get_interpolated_absorption(molecule: str, exponent: float, 
                                wave: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get absorption transmission by interpolating from precomputed grid.
    
    Parameters
    ----------
    molecule : str
        Molecule name ('H2O', 'CH4', 'CO2', 'O2')
    exponent : float
        Absorption exponent
    wave : np.ndarray, optional
        Target wavelength grid. If None, returns on reference grid.
        
    Returns
    -------
    transmission : np.ndarray
        Interpolated transmission
    weights : np.ndarray
        Interpolated weights
    """
    global _precomputed_grid, _precomputed_waveref
    
    if _precomputed_grid is None:
        raise RuntimeError("Precomputed grid not loaded. Call precompute_absorption_grid() first.")
    
    mol_grid = _precomputed_grid[molecule]
    expo_values = sorted(mol_grid.keys())
    
    # Clamp exponent to grid range
    expo_min, expo_max = expo_values[0], expo_values[-1]
    expo_clamped = max(expo_min, min(expo_max, exponent))
    
    # Find bracketing exponents
    expo_below = expo_values[0]
    expo_above = expo_values[-1]
    for ev in expo_values:
        if ev <= expo_clamped:
            expo_below = ev
        if ev >= expo_clamped:
            expo_above = ev
            break
    
    # Get bracketing data
    data_below = mol_grid[expo_below]
    data_above = mol_grid[expo_above]
    
    # Log-space interpolation weight (correct for log-uniform grid)
    # t = (log(expo) - log(expo_below)) / (log(expo_above) - log(expo_below))
    if expo_above == expo_below:
        t = 0.0
    else:
        t = (np.log(expo_clamped) - np.log(expo_below)) / (np.log(expo_above) - np.log(expo_below))
    
    # Interpolate in exponent space on reference grid FIRST
    # This is more efficient than splining twice: we do weighted mean on the 
    # master grid, then a single spline to the observation grid
    trans_interp = (1 - t) * data_below['transmission'] + t * data_above['transmission']
    weights_interp = (1 - t) * data_below['weights'] + t * data_above['weights']
    
    # Resample to target wavelength grid if different from reference
    # The precomputed grid is on waveref (master grid), observation wavelengths
    # vary slightly due to barycentric correction and drift
    if wave is not None and not np.allclose(wave, _precomputed_waveref):
        # Single spline interpolation to observation wavelength grid
        trans_out = np.zeros_like(wave)
        weights_out = np.zeros_like(wave)
        for iord in range(wave.shape[0]):
            trans_out[iord] = ius(_precomputed_waveref[iord], trans_interp[iord], 
                                  k=1, ext=0)(wave[iord])
            weights_out[iord] = ius(_precomputed_waveref[iord], weights_interp[iord], 
                                    k=1, ext=0)(wave[iord])
        return trans_out, weights_out
    
    return trans_interp, weights_interp


def is_precomputed_ready() -> bool:
    """Check if precomputed grid is loaded."""
    return _precomputed_grid is not None


def construct_abso(wave: np.ndarray,
                  expos: List[float],
                  all_abso: Optional[np.ndarray] = None,
                  instrument: str = DEFAULT_INSTRUMENT,
                  apply_final_mask: bool = False,
                  use_waveref: bool = False) -> np.ndarray:
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
    apply_final_mask : bool, optional
        If True, apply hard mask where weight < 0.5 → NaN.
        Use False during optimization (smooth weights only),
        True for final spectrum output. Default: False
    use_waveref : bool, optional
        If True, assume wave is identical to waveref and skip spline
        resampling (much faster). Use during optimization when
        wavelengths don't change. Default: False

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

    The smooth weights are always computed and stored in construct_abso.last_weights.
    When apply_final_mask=True, pixels with weight < 0.5 are set to NaN.
    """
    # Load molecular absorptions if not provided
    if all_abso is None:
        params = get_user_params(instrument)

        if instrument == 'NIRPS':
            transm_file = 'LaSilla_NIRPS_tapas.fits'
        elif instrument == 'SPIROU':
            transm_file = 'MaunaKea_tapas.fits'
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

    # Per-molecule convolution and weighting
    # We convolve each molecule separately to compute per-molecule depth weights
    # Use precomputed grid if available (much faster), else fall back to caching
    trans2 = np.ones_like(wave)
    combined_weights = np.ones(wave.shape, dtype=np.float64)

    if is_precomputed_ready():
        # Fast path: interpolate from precomputed grid
        # Pass wave=None if use_waveref to skip expensive spline resampling
        wave_arg = None if use_waveref else wave
        for i, molecule in enumerate(MOLECULES):
            mol_trans_conv, mol_weights = get_interpolated_absorption(
                molecule, expos[i], wave_arg
            )
            trans2 *= mol_trans_conv
            combined_weights *= mol_weights
    else:
        # Fallback: compute convolutions with caching
        global _conv_cache

        for i, molecule in enumerate(MOLECULES):
            # Round exponent to 3 decimals for cache key
            expo_key = round(expos[i], 3)
            cache_key = (i, expo_key)

            # Check if we have this convolution cached
            if cache_key in _conv_cache:
                mol_trans_conv = _conv_cache[cache_key]
            else:
                # Compute this molecule's transmission with exponent
                mol_trans = all_abso[i] ** expos[i]

                # Convolve with instrumental resolution
                mol_trans_conv = variable_res_conv(wave, mol_trans, E2DS_FWHM, E2DS_EXPO)

                # Cache the result (with size limit)
                if len(_conv_cache) >= _conv_cache_max_size:
                    # Remove oldest entry (first key)
                    oldest_key = next(iter(_conv_cache))
                    del _conv_cache[oldest_key]
                _conv_cache[cache_key] = mol_trans_conv

            # Get molecule-specific parameters and compute smooth weights
            mol_params = get_molecule_params(molecule)

            # Compute weights per order (row) since get_transmission_weights expects 1D
            for iord in range(wave.shape[0]):
                order_weights = get_transmission_weights(
                    mol_trans_conv[iord],
                    depth_max=mol_params['depth_max'],
                    depth_saturated=mol_params['depth_saturated'],
                    reject_saturated=mol_params['reject_saturated']
                )
                # Multiply weights: any low weight in any molecule reduces overall weight
                combined_weights[iord] *= order_weights

            # Multiply into combined transmission
            trans2 *= mol_trans_conv

    # Store weights as an attribute for use in optimization
    # We return trans2 but the weights are accessible via construct_abso.last_weights
    construct_abso.last_weights = combined_weights

    # Apply final mask if requested (for output spectra, not during optimization)
    if apply_final_mask:
        trans2[combined_weights < 0.5] = np.nan

    return trans2

# Initialize the attribute
construct_abso.last_weights = None


# ============================================================================
# Exponent Optimization
# ============================================================================

def optimize_exponents(wave: np.ndarray,
                      sp: np.ndarray,
                      airmass: float,
                      fixed_exponents: Optional[List] = None,
                      guess: Optional[List] = None,
                      blaze: Optional[np.ndarray] = None,
                      instrument: str = DEFAULT_INSTRUMENT) -> List[float]:
    """
    Optimize atmospheric absorption exponents.

    Finds the best-fit exponents by minimizing the gradient of the
    telluric-corrected spectrum (weighted by per-molecule depth thresholds).

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
    blaze : np.ndarray, optional
        Blaze function (same shape as sp). If provided, used to down-weight
        low-flux regions where normalized blaze < 0.3
    instrument : str
        Instrument name

    Returns
    -------
    expo_optimal : list of float
        Optimized exponents [H2O, CH4, CO2, O2]

    Notes
    -----
    The optimization minimizes:
        σ(∇(sp/T) * w * T)

    where:
    - T is transmission
    - w is per-molecule weighting from telluric_config.yaml (smooth weights
      based on depth_max, depth_saturated, reject_saturated thresholds)
    - ∇ is spatial gradient

    The smooth weights prevent convergence oscillations that would occur
    with hard binary masking. The gradient should be flat if the telluric
    correction is perfect.
    
    Also applies:
    - Blaze weighting: normalized blaze per order, set to 0 where < 0.3
    - Outlier rejection: sigmoid falloff from 4-8 sigma (0.5 at 6 sigma)
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
    weights_ref = construct_abso.last_weights

    # Wavelength mask for fitting region
    params = get_user_params(instrument)
    wave_fit = params['wave_fit']
    wave_mask_inv = ~((wave >= wave_fit[0]) & (wave <= wave_fit[1]))

    # Pre-compute pixel-to-pixel RMS (vectorized)
    pix2pixrms = np.nanmedian(np.abs(np.diff(sp, axis=1)), axis=1)
    pix2pixrms_30 = 30.0 * pix2pixrms[:, np.newaxis]

    # Compute blaze weights: normalize each order by 90th percentile, zero where < 0.3
    if blaze is not None:
        blaze_weights = np.zeros_like(blaze)
        for iord in range(blaze.shape[0]):
            blaze_90 = np.nanpercentile(blaze[iord], 90)
            if blaze_90 > 0:
                blaze_norm = blaze[iord] / blaze_90
                blaze_weights[iord] = np.where(blaze_norm >= 0.3, blaze_norm, 0.0)
            else:
                blaze_weights[iord] = 0.0
    else:
        blaze_weights = None

    # Build indices for variable exponents
    if fixed_exponents is None:
        var_indices = [0, 1, 2, 3]
    else:
        var_indices = [i for i in range(4) if fixed_exponents[i] is None]

    molecules_names = ['H2O', 'CH4', 'CO2', 'O2']
    free_molecules = [molecules_names[i] for i in var_indices]
    tprint(f'  Starting exponent optimization for {free_molecules}...')

    # Cache for objective function to avoid double computation in callback
    _obj_cache = {'last_x': None, 'last_val': None, 'last_expos': None}

    def optimize_expo(variable_expos):
        """Objective function to minimize."""
        # Check cache first (callback often calls with same x as optimizer just evaluated)
        x_tuple = tuple(variable_expos)
        if _obj_cache['last_x'] == x_tuple:
            return _obj_cache['last_val']

        # Reconstruct full exponents list
        if fixed_exponents is None:
            expos = list(variable_expos)
        else:
            expos = list(fixed_exponents)
            for j, i in enumerate(var_indices):
                expos[i] = variable_expos[j]

        # Compute transmission model (also updates construct_abso.last_weights)
        trans2 = construct_abso(wave, expos=expos, all_abso=all_abso,
                               instrument=instrument)
        mol_weights = construct_abso.last_weights

        # Corrected spectrum
        corr = sp / trans2
        corr[wave_mask_inv] = np.nan

        # Compute gradient
        grad = np.gradient(corr, axis=1)
        
        # Build combined weights with blaze and outlier rejection
        combined_weights = mol_weights.copy()
        
        # Apply blaze weights (zero where blaze < 0.3 of 90th percentile)
        if blaze_weights is not None:
            combined_weights *= blaze_weights
        
        # Per-order outlier rejection using sigma clipping with sigmoid falloff
        # Compute 1-sigma from 16th-84th percentile of non-masked gradient values
        for iord in range(wave.shape[0]):
            # Get non-masked values (mol_weights > 0.5)
            valid_mask = mol_weights[iord] > 0.5
            if np.sum(valid_mask) < 10:
                # Not enough valid points, zero this order
                combined_weights[iord] = 0.0
                continue
            
            grad_valid = grad[iord][valid_mask]
            p16, p84 = np.nanpercentile(grad_valid, [16, 84])
            sigma_order = (p84 - p16) / 2.0  # 1-sigma estimate
            
            if sigma_order <= 0 or not np.isfinite(sigma_order):
                continue
            
            # Compute deviation in sigma units
            median_grad = np.nanmedian(grad_valid)
            deviation = np.abs(grad[iord] - median_grad) / sigma_order
            
            # Sigmoid falloff: 0.5 at 6 sigma, width 1
            # weight = 1 / (1 + exp((deviation - 6) / 1))
            # This gives ~1 for <4 sigma, 0.5 at 6 sigma, ~0 for >8 sigma
            outlier_weight = 1.0 / (1.0 + np.exp((deviation - 6.0) / 1.0))
            combined_weights[iord] *= outlier_weight

        # Objective: weighted standard deviation of gradient
        val_sum = weighted_nanstd(grad, weights=combined_weights)

        # Cache result
        _obj_cache['last_x'] = x_tuple
        _obj_cache['last_val'] = val_sum
        _obj_cache['last_expos'] = expos

        return val_sum

    # Callback for progress tracking
    iteration_count = [0]  # Use list to allow modification in nested function
    last_obj = [None]

    def progress_callback(xk):
        """Print optimization progress at each iteration (overwrites line)."""
        iteration_count[0] += 1
        current_obj = optimize_expo(xk)

        # Build current exponents for display
        if fixed_exponents is None:
            current_expos = list(xk)
        else:
            current_expos = list(fixed_exponents)
            for j, i in enumerate(var_indices):
                current_expos[i] = xk[j]

        # Format exponent values
        expo_str = ', '.join([f'{molecules_names[i]}={current_expos[i]:.4f}'
                              for i in var_indices])

        # Show improvement
        if last_obj[0] is not None:
            delta = current_obj - last_obj[0]
            delta_str = f' Δ={delta:+.1e}'
        else:
            delta_str = ''

        # Print on same line using carriage return
        msg = f'    Iter {iteration_count[0]:3d}: obj={current_obj:.3e}{delta_str} | {expo_str}'
        print(f'\r{msg}', end='', flush=True)
        last_obj[0] = current_obj

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

    # Print initial state
    initial_obj = optimize_expo(x0)
    expo_str = ', '.join([f'{molecules_names[var_indices[j]]}={x0[j]:.4f}'
                          for j in range(len(var_indices))])
    tprint(f'    Initial: obj={initial_obj:.4e} | {expo_str}', color='blue')

    # Optimize with progress callback
    with np.errstate(invalid='ignore'):
        result = minimize(
            optimize_expo,
            x0=x0,
            bounds=bounds,
            method=config['method'],
            tol=config['tolerance'],
            callback=progress_callback
        )

    # Print final result (newline first to clear the progress line)
    print()  # Move to new line after progress updates
    final_obj = optimize_expo(result.x)
    tprint(f'    Final: obj={final_obj:.3e} ({iteration_count[0]} iter, converged={result.success})',
           color='green')

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

def mask_o2(wave: np.ndarray, instrument: str = DEFAULT_INSTRUMENT) -> np.ndarray:
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


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Configuration
    'user_params',
    'get_blaze',
    'get_header_transm',
    'load_telluric_config',
    'get_molecule_params',
    'E2DS_FWHM',
    'E2DS_EXPO',
    'BLAZE',
    'DEFAULT_INSTRUMENT',

    # FITS I/O
    'getdata_safe',
    'getheader_safe',

    # Sky reconstruction
    'sky_pca_fast',
    'sky_pca',

    # Velocity
    'gauss',
    'get_velo',

    # Headers
    'update_header',
    'hotstar',

    # Airmass
    'accurate_airmass',
    'delay_since_sunset',

    # Utilities
    'weight_fall',
    'savgol_filter_nan_fast',

    # Transmission masking/weighting
    'get_valid_transmission_mask',
    'get_valid_molecule_mask',
    'get_transmission_weights',
    'get_molecule_weights',
    'super_gauss_fast',
    'variable_res_conv',

    # Absorption
    'construct_abso',
    'mask_o2',

    # Optimization
    'optimize_exponents',
]
