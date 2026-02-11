"""
Configuration module for tellu_tools.

This module centralizes all configuration parameters, paths, and constants
used by the telluric correction tools.

Author: Refactored from tellu_tools.py
Date: 2026-01-12
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any
from astropy.io import fits
import yaml


# ============================================================================
# Timestamped Logging
# ============================================================================

# ANSI color codes
class Colors:
    GREEN = '\033[38;5;28m'    # Darker green (256-color palette)
    BLUE = '\033[94m'
    ORANGE = '\033[38;5;208m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'


def tprint(*args, color='green', **kwargs):
    """
    Timestamped colored print function.
    
    Prints messages with a timestamp prefix and optional color in the format:
    HH:MM:SS.mmm | message
    
    Parameters
    ----------
    *args : positional arguments
        Arguments to print (same as print())
    color : str
        Color of the message: 'green' (default), 'blue', 'orange'
    **kwargs : keyword arguments  
        Keyword arguments passed to print() (end, sep, file, flush)
    
    Examples
    --------
    >>> tprint("Loading calibration files")
    12:34:56.789 | Loading calibration files
    
    >>> tprint("Processing order", 42, color='blue')
    12:34:56.790 | Processing order 42
    
    >>> tprint("Warning: high airmass", color='orange')
    12:34:56.791 | Warning: high airmass
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    message = " ".join(str(arg) for arg in args)
    
    # Select color
    color_map = {
        'green': Colors.GREEN,
        'blue': Colors.BLUE,
        'orange': Colors.ORANGE,
        'cyan': Colors.CYAN,
        'magenta': Colors.MAGENTA,
        'yellow': Colors.YELLOW,
        'red': Colors.RED,
    }
    color_code = color_map.get(color.lower(), Colors.GREEN)
    
    # Color only applies to the message AFTER the timestamp separator
    print(f"{timestamp} | {color_code}{message}{Colors.RESET}", **kwargs)
    # Flush to ensure immediate output
    sys.stdout.flush()


# ============================================================================
# Instrument Configuration
# ============================================================================

SUPPORTED_INSTRUMENTS = ['NIRPS', 'SPIROU']

# Wavelength fit ranges by instrument (in nm)
WAVELENGTH_FIT_RANGES = {
    'NIRPS': [980, 1800],
    'SPIROU': [980, 2400],
}

# Molecules to model
MOLECULES = ['H2O', 'CH4', 'CO2', 'O2']

# Physical constants
SPEED_OF_LIGHT = 299792.458  # km/s


# ============================================================================
# Path Configuration
# ============================================================================

# Cache for project path (to avoid repeated detection messages)
_cached_project_path = None

def get_project_path() -> str:
    """
    Determine project path based on the computing environment.
    
    Reads machine configurations from batch_config.yaml and detects
    which machine we're on by checking if detect_path exists.

    Returns
    -------
    project_path : str
        Root path to project data
    """
    global _cached_project_path
    
    # Return cached result if available
    if _cached_project_path is not None:
        return _cached_project_path
    
    # Load machine configurations from batch_config.yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'batch_config.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        machines = config.get('machines', {})
        for machine_name, machine_config in machines.items():
            detect_path = machine_config.get('detect_path', '')
            if detect_path and os.path.exists(detect_path):
                _cached_project_path = machine_config.get('project_path', detect_path)
                tprint(f"Detected machine: {machine_name}", color='blue')
                return _cached_project_path
    
    # Fallback: try to determine from script location
    project_path = os.path.dirname(script_dir)
    if os.path.exists(os.path.join(project_path, 'calib_NIRPS')) or \
       os.path.exists(os.path.join(project_path, 'calib_SPIROU')):
        tprint("Using project path from script location", color='orange')
        _cached_project_path = project_path
        return _cached_project_path
    
    raise RuntimeError(
        "Could not determine project path. Please add your machine "
        "configuration to batch_config.yaml"
    )


def get_user_params(instrument: str = 'NIRPS') -> Dict[str, Any]:
    """
    Get user parameters including paths and processing settings.

    Parameters
    ----------
    instrument : str
        Instrument name ('NIRPS' or 'SPIROU')

    Returns
    -------
    params : dict
        Dictionary with project_path, doplot, knee, wave_fit
    """
    if instrument not in SUPPORTED_INSTRUMENTS:
        raise ValueError(f"Unsupported instrument: {instrument}. "
                        f"Must be one of {SUPPORTED_INSTRUMENTS}")

    project_path = get_project_path()
    wave_fit = WAVELENGTH_FIT_RANGES[instrument]

    return {
        'project_path': project_path,
        'doplot': False,  # Set to True for diagnostic plots
        'knee': 0.3,      # Absorption threshold for masking
        'wave_fit': wave_fit,
    }


# ============================================================================
# Sky PCA Configuration
# ============================================================================

SKY_PCA_CONFIG = {
    'bands': [
        (950, 1400, 'Y+J'),
        (1400, 1900, 'H'),
    ],
    'max_iterations': 200,
    'ftol': 1e-8,
    'gtol': 1e-3,
}


# ============================================================================
# Velocity Optimization Configuration
# ============================================================================

VELOCITY_CONFIG = {
    'dv_step': 0.5,          # km/s, velocity step
    'dv_amp_default': 200,   # km/s, default search range
    'coarse_step': 10,       # Coarse search every N steps
    'fine_range': 20,        # km/s, fine search range around peak
    'gaussian_p0': [None, None, 5.0, 0, 2],  # Initial guess for Gaussian fit
}


# ============================================================================
# Exponent Optimization Configuration
# ============================================================================

EXPONENT_OPT_CONFIG = {
    'method': 'Nelder-Mead',
    'tolerance': 5e-4,
    'h2o_bounds': (0.001, 20),      # H2O exponent bounds
    'airmass_tolerance': 0.1,        # ±10% airmass tolerance
    'knee_factor': 0.5,              # Factor for relevant absorption threshold
    'max_absorption': 0.95,          # Maximum transmission considered
    'outlier_threshold': 30.0,       # Sigma threshold for outlier rejection
}


# ============================================================================
# Template Configuration
# ============================================================================

TEMPLATE_CONFIG = {
    'temperature_grid': list(range(3000, 6500, 500)),  # Kelvin
    'temp_min': 3000,    # K
    'temp_max': 6000,    # K
    'temp_step': 500,    # K
    'hot_star_list': [
        'HD195094', 'HR1903', 'HR4023', 'HR3131', 'HR6743',
        'HR7590', 'HR8709', 'HR9098', 'HR3117', 'HR3314', 'HR4467'
    ],
}


# ============================================================================
# Airmass Calculation Configuration
# ============================================================================

AIRMASS_CONFIG = {
    'R_earth': 6371.0,  # km, Earth radius
    'H_atmo': 8.43,     # km, atmospheric scale height
}


# ============================================================================
# Convolution Configuration
# ============================================================================

CONVOLUTION_CONFIG = {
    'kernel_threshold': 1e-4,  # Amplitude threshold to stop convolution
    'fwhm_scale': 3.0,         # FWHM scaling factor
    'range_scan_scale': 20,    # Scan range scaling factor
}


# ============================================================================
# Calibration File Paths
# ============================================================================

def get_calib_paths(instrument: str, project_path: str = None) -> Dict[str, str]:
    """
    Get paths to calibration files for a given instrument.

    Parameters
    ----------
    instrument : str
        Instrument name
    project_path : str, optional
        Root project path (if None, uses get_project_path())

    Returns
    -------
    paths : dict
        Dictionary with calibration file paths
    """
    if project_path is None:
        project_path = get_project_path()

    if instrument == 'NIRPS':
        return {
            'fwhm_file': os.path.join(
                project_path,
                'calib_NIRPS/C7A164F31A_pp_e2dsff_A_waveref_res_e2ds_A.fits'
            ),
            'expo_ext': 'E2DS_EXPO',
            'fwhm_ext': 'E2DS_FWHM',
            'blaze_file': os.path.join(
                project_path,
                'calib_NIRPS/07337C08CA_pp_blaze_A.fits'
            ),
            'tapas_file': os.path.join(project_path, 'LaSilla_tapas.fits'),
            'waveref_file': os.path.join(project_path, 'calib_NIRPS/waveref.fits'),
        }

    elif instrument == 'SPIROU':
        return {
            'fwhm_file': os.path.join(
                project_path,
                'calib_SPIROU/3444961B5Da_pp_e2dsff_AB_waveref_res_e2ds_AB.fits'
            ),
            'expo_ext': 'E2DS_EXPO',
            'fwhm_ext': 'E2DS_FWHM',
            'blaze_file': os.path.join(
                project_path,
                'calib_SPIROU/5ABA102B11f_pp_blaze_AB.fits'
            ),
            'tapas_file': os.path.join(project_path, 'MaunaKea_tapas.fits'),
            'waveref_file': os.path.join(project_path, 'calib_SPIROU/waveref.fits'),
        }

    else:
        raise ValueError(f"Unknown instrument: {instrument}")


# ============================================================================
# Header Key Mappings
# ============================================================================

HEADER_KEYS = {
    'NIRPS': {
        'airmass_start': 'ESO TEL AIRM START',
        'airmass_end': 'ESO TEL AIRM END',
        'pressure_start': 'ESO TEL AMBI PRES START',
        'pressure_end': 'ESO TEL AMBI PRES END',
        'temperature': 'ESO TEL AMBI TEMP',
        'humidity': 'ESO TEL AMBI RHUM',
        'snr_ref': 'EXTSN060',  # SNR at ~1.60 micron
    },
    'SPIROU': {
        'airmass': 'AIRMASS',
        'pressure': 'PRESSURE',
        'temperature': 'TEMPERAT',
        'humidity': 'RELHUMID',
        'snr_ref': 'EXTSN042',  # SNR at ~1.60 micron
    },
}


# ============================================================================
# Processing Defaults
# ============================================================================

PROCESSING_DEFAULTS = {
    'lowpass_filter_size': 101,
    'savgol_window': 9,
    'savgol_polyorder': 2,
    'robust_weight_threshold': 1e-4,
    'weight_transition_factor': 4.0,
    'min_valid_fraction': 0.3,
}


# ============================================================================
# Logging Configuration
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
}


# ============================================================================
# Validation Functions
# ============================================================================

def validate_instrument(instrument: str) -> None:
    """Validate instrument name."""
    if instrument not in SUPPORTED_INSTRUMENTS:
        raise ValueError(
            f"Invalid instrument: {instrument}. "
            f"Supported instruments: {SUPPORTED_INSTRUMENTS}"
        )


def validate_molecules(molecules: list) -> None:
    """Validate molecule list."""
    for mol in molecules:
        if mol not in MOLECULES:
            raise ValueError(
                f"Invalid molecule: {mol}. "
                f"Supported molecules: {MOLECULES}"
            )


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration to validate

    Raises
    ------
    ValueError
        If configuration is invalid
    """
    required_keys = ['project_path', 'doplot', 'knee', 'wave_fit']

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")

    if not os.path.exists(config['project_path']):
        raise ValueError(f"Project path does not exist: {config['project_path']}")

    if not isinstance(config['doplot'], bool):
        raise ValueError("doplot must be boolean")

    if not (0 < config['knee'] < 1):
        raise ValueError("knee must be between 0 and 1")

    if len(config['wave_fit']) != 2:
        raise ValueError("wave_fit must be a list of 2 elements [min, max]")


# ============================================================================
# Export commonly used configurations
# ============================================================================

# Default user parameters (legacy compatibility)
def user_params(instrument: str = 'NIRPS') -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    Equivalent to get_user_params().
    """
    return get_user_params(instrument)


if __name__ == '__main__':
    # Example usage and validation
    print("Telluric Tools Configuration")
    print("=" * 70)

    for instrument in SUPPORTED_INSTRUMENTS:
        print(f"\n{instrument} Configuration:")
        print("-" * 70)

        params = get_user_params(instrument)
        for key, value in params.items():
            print(f"  {key}: {value}")

        print(f"\nCalibration paths:")
        calib_paths = get_calib_paths(instrument)
        for key, path in calib_paths.items():
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {exists} {key}: {path}")

    print("\n" + "=" * 70)
    print("Configuration module loaded successfully")
