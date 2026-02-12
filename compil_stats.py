"""
compil_stats.py - Telluric Parameter Statistical Compilation

This script compiles and analyzes telluric absorption parameters from a collection
of hot star observations to derive empirical scaling laws for atmospheric molecules.

Purpose:
--------
1. Aggregate telluric fit results from individual observations into a master table
2. Fit empirical models for CO2, CH4, and O2 absorption as functions of:
   - Airmass (optical path length through atmosphere)
   - Temperature (affects line broadening and molecular densities)
   - Pressure (affects line broadening via collisional broadening)
   - Time (seasonal variations for CO2/CH4 due to biosphere uptake/release)
3. Identify and flag outliers using iterative sigma-clipping
4. Generate quality control diagnostics and validation plots
5. Output fitted parameters for use in predict_abso.py absorption predictions

Physical Background:
-------------------
- CO2: Shows ~2% annual cycle (Northern Hemisphere biosphere) + secular ~2 ppm/year increase
- CH4: Shows seasonal variation + secular increase from anthropogenic sources  
- O2: Constant mixing ratio (20.95%) but requires careful fluorescence filtering
  (Meinel bands excited by twilight UV - excluded via O2_VALID mask)
- H2O: Highly variable, fitted per-observation rather than from statistics

Outputs:
--------
- params_fit_tellu_{INSTRUMENT}.csv: Fitted scaling parameters for each molecule
- main_absorber_{INSTRUMENT}.fits: Map of dominant absorber at each wavelength
- o2_airmass_hotstar_check.png: QC diagnostic plot for O2 filtering

Author: Etienne Artigau
"""

from astropy.io import fits
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from astropy.table import Table
from scipy.optimize import curve_fit
from tellu_tools import construct_abso, optimize_exponents, hotstar, sky_pca_fast, load_telluric_config
from tellu_tools_config import tprint, get_user_params
from aperocore import math as mp
from tqdm import tqdm
import os
import sys
import warnings

# Suppress FITS warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*Card is too long.*')
warnings.filterwarnings('ignore', message='.*VerifyWarning.*')

# =============================================================================
# Global State
# =============================================================================

# Track which paper figures have been generated (prevents duplicates during reruns)
_paper_figure_done = {
    'fig7': False,      # QC diagnostics (O2 airmass, outlier flagging)
    'fig_o2': False,    # O2 airmass fit with model
    'fig_co2': False,   # CO2 temporal fit with seasonal cycle
    'fig_ch4': False,   # CH4 temporal fit with seasonal cycle
}


# =============================================================================
# Configuration Functions
# =============================================================================

def get_paper_figures_config(instrument: str = 'NIRPS'):
    """
    Get paper figures configuration from YAML config file.
    
    Checks if paper figure generation is enabled and returns the output directory.
    Creates the output directory if it doesn't exist.
    
    Parameters
    ----------
    instrument : str
        Instrument name ('NIRPS' or 'SPIROU')
        
    Returns
    -------
    tuple : (enabled: bool, output_dir: str or None)
        Whether paper figures are enabled and path to output directory
    """
    config = load_telluric_config()
    paper_config = config.get('paper_figures', {})
    enabled = paper_config.get('enabled', False)
    
    if not enabled:
        return False, None
    
    params = get_user_params(instrument)
    project_path = params['project_path']
    output_dir = os.path.join(project_path, paper_config.get('output_dir', 'paper_figures'))
    os.makedirs(output_dir, exist_ok=True)
    return True, output_dir

# =============================================================================
# Fitting Models
# =============================================================================

def anthropic(mjds_airmass, *params):
    """
    Full atmospheric molecule model with temporal and environmental dependencies.
    
    Models molecule concentration as a function of time (secular trend + seasonal),
    airmass, temperature, and pressure. Used for CO2 and CH4 which show clear
    temporal variations due to biosphere activity and anthropogenic emissions.
    
    Model: VMR(t) = [slope*t + intercept + amplitude*cos(2π*t/year + phase)] 
                    × airmass^α × temperature^β × pressure^γ
    
    Parameters
    ----------
    mjds_airmass : list
        [0] mjds : Modified Julian Date (time of observation)
        [1] airmass : Optical path length through atmosphere (1.0 at zenith)
        [2] temperature : Normalized temperature (T/273.15 K)
        [3] pressure_norm : Normalized pressure (P/P_standard)
    *params : tuple of 7 floats
        [0] slope : Secular trend (ppm/day) - e.g., CO2 increase ~2 ppm/year
        [1] intercept : Base concentration at t=0 (ppm)
        [2] amp : Amplitude of annual sinusoidal variation (ppm)
        [3] phase : Phase offset of annual cycle (radians)
        [4] airmass_exp : Power law exponent for airmass dependence
        [5] temperature_exp : Power law exponent for temperature dependence
        [6] pressure_exp : Power law exponent for pressure dependence
        
    Returns
    -------
    ndarray : Predicted volume mixing ratio or equivalent absorption measure
    """
    mjds = mjds_airmass[0]
    airmass = mjds_airmass[1]
    temperature = mjds_airmass[2]
    pressure_norm = mjds_airmass[3]
    
    # Temporal component: linear trend + annual sinusoidal variation
    # The 365.24 accounts for the tropical year length
    y = (params[0]*mjds + params[1] + params[2]*np.cos(2.0*np.pi*(mjds % 365.24)/365.24 + params[3]))
    
    # Environmental scaling: power laws for airmass, temperature, and pressure
    return y * airmass**params[4] * (temperature)**params[5] * (pressure_norm)**params[6]

def simple_scaling(airmass_temperature_pressure, *params):
    """
    Simple power-law scaling model without temporal dependence.
    
    Used for O2 which has a constant mixing ratio (20.95%) and doesn't vary
    with season or secular trends. Only depends on optical path length
    (airmass) and atmospheric state (temperature, pressure).
    
    Model: O2_airmass = airmass^α × temperature^β × pressure^γ + offset
    
    Parameters
    ----------
    airmass_temperature_pressure : list
        [0] airmass : Optical path length (1.0 at zenith, increases with zenith angle)
        [1] temperature : Normalized temperature (T/273.15 K)
        [2] pressure_norm : Normalized pressure
    *params : tuple of 4 floats
        [0] airmass_exp : Power law exponent for airmass (should be ~1.0)
        [1] temperature_exp : Power law exponent for temperature dependence
        [2] pressure_exp : Power law exponent for pressure dependence  
        [3] intercept : Additive offset (should be close to 0)
        
    Returns
    -------
    ndarray : Predicted O2 effective airmass
    
    Notes
    -----
    The O2 airmass parameter from fits represents the effective optical depth
    normalized to a reference. For well-behaved data, params[0] ≈ 1.0 and
    params[3] ≈ 0.0, indicating O2 absorption scales linearly with airmass.
    """
    airmass = airmass_temperature_pressure[0]
    temperature = airmass_temperature_pressure[1]
    pressure_norm = airmass_temperature_pressure[2]

    return airmass**params[0] * (temperature)**params[1] * (pressure_norm)**params[2] + params[3]

def accurate_airmass(z):
    """
    Calculate accurate airmass accounting for Earth's curvature.
    
    Uses the spherical shell model rather than plane-parallel approximation.
    Important for zenith angles > 60° where sec(z) approximation fails.
    
    The formula accounts for:
    - Earth's spherical geometry (radius R_earth = 6371 km)
    - Finite atmospheric scale height (H_atmo = 8.43 km, for homogeneous atmosphere)
    
    Parameters
    ----------
    z : float or ndarray
        Zenith angle in degrees (0° = overhead, 90° = horizon)
        
    Returns
    -------
    float or ndarray : Airmass (1.0 at zenith, ~38 at horizon)
    
    Notes
    -----
    At z=0°: airmass = 1.0 (looking straight up)
    At z=60°: airmass ≈ 2.0 (sec(60°) = 2)
    At z=80°: airmass ≈ 5.6 (deviation from sec(z) = 5.76 becomes noticeable)
    At z=90°: airmass ≈ 38 (finite, unlike sec(90°) = ∞)
    """
    R_earth = 6371.0  # Earth radius in km
    H_atmo = 8.43     # Atmospheric scale height in km (pressure e-folding height)
    z_rad = np.radians(z)
    sec_z = 1.0 / np.cos(z_rad)
    
    # Spherical shell geometry correction
    am = np.sqrt((R_earth / (R_earth + H_atmo))**2 * sec_z**2 
                 - (R_earth / (R_earth + H_atmo))**2 + 1.0)
    return am

# =============================================================================
# Main Script Configuration
# =============================================================================

# Dictionary to accumulate all fitted parameters for output
tbl_params_fit = dict()

# Instrument selection: 'NIRPS' (near-IR) or 'SPIROU' (near-IR, different site)
# Each instrument has different wavelength coverage and atmospheric conditions
#instrument = 'SPIROU'
instrument = 'NIRPS'

# Sigma-clipping threshold for outlier rejection in iterative fits
# Points deviating more than sigma_cut × robust_std are removed
sigma_cut = 4.0

# Get the correct project path for this environment
params = get_user_params(instrument)
project_path = params['project_path']

outname = 'params_fit_tellu_'+instrument+'.csv'

# Check if calibration files exist
calib_path = os.path.join(project_path, f'calib_{instrument}/waveref.fits')
if not os.path.exists(calib_path):
    tprint(f'ERROR: Calibration file not found: {calib_path}', color='red')
    tprint(f'Project path: {project_path}', color='blue')
    tprint(f'Please run the sync step first: bash sync', color='yellow')
    sys.exit(1)

waveref = fits.getdata(calib_path)

big_table_file = os.path.join(project_path, f'tellu_fit_{instrument}/big_table.csv')

# =============================================================================
# Data Loading: Build or Load Master Observation Table
# =============================================================================

# Get list of all telluric transmission fit files (one per observation)
files = glob.glob(os.path.join(project_path, f'tellu_fit_{instrument}/trans_*.fits'))
files = sorted(files)
n_files = len(files)
tprint(f'Found {n_files} trans_*.fits files in {project_path}/tellu_fit_{instrument}/', color='green')

# Check if cached big_table.csv exists and is up-to-date
# Rebuild if: (1) doesn't exist, or (2) row count doesn't match file count (stale cache)
rebuild_table = False
if not os.path.exists(big_table_file):
    tprint('big_table.csv not found, building from scratch...', color='yellow')
    rebuild_table = True
else:
    # Check if cache is stale by comparing row count to file count
    cached_tbl = Table.read(big_table_file, format='csv')
    n_cached = len(cached_tbl)
    if n_cached != n_files:
        tprint(f'big_table.csv is stale: {n_cached} rows but {n_files} files exist', color='yellow')
        tprint('Rebuilding big_table.csv...', color='yellow')
        rebuild_table = True
    else:
        tprint(f'big_table.csv is up-to-date ({n_cached} rows)', color='green')

# Build the big_table from scratch if needed, otherwise use cached version
# This table contains all relevant header keywords from each observation
if rebuild_table:

    # Header keywords to extract from each telluric fit file:
    # - Environmental: AIRMASS, PRESSURE, TEMPERAT, HUMIDITY, NORMPRES
    # - Timing: SUNSETD (hours since sunset), MJDMID, DRSSUNEL, EXPTIME
    # - Fitted quantities: H2O_CV, CO2_VMR, CH4_VMR, O2_AIRM (molecule abundances)
    # - Fitted exponents: EXPO_H2O, EXPO_CH4, EXPO_CO2, EXPO_O2
    # - Quality: SNR_REF
    # - Target: DRSOBJN (object name for hot star filtering)
    keys = ['AIRMASS', 'PRESSURE','TEMPERAT','SUNSETD','HUMIDITY','NORMPRES',
            'H2O_CV','CO2_VMR','CH4_VMR','O2_AIRM','EXPO_H2O',
            'EXPO_CH4','EXPO_CO2','EXPO_O2','MJDMID','DRSSUNEL','SNR_REF',
            'EXPTIME','NORMPRES','ACCAIRM','DRSOBJN']

    tbl = Table()
    tbl['FILENAME'] = files
    for key in keys:
        tbl[key] = None

    for file in tqdm(files, desc='Reading headers', leave = False, unit='files'):
        h = fits.getheader(file)
        h = hotstar(h)
        for key in keys:
            try:
                tbl[key][tbl['FILENAME'] == file] = h[key]
            except:
                tprint(f'Warning: key {key} not found in file {file}', color='orange')
    tbl.write(big_table_file, format='csv', overwrite=True)
else:
    # Use the already-loaded cached table
    tbl = cached_tbl
    keys = [col for col in tbl.colnames if col != 'FILENAME']

# Flag morning observations (>5 hours after sunset = past midnight)
# Morning observations are preferred for O2 fitting because Meinel band
# fluorescence (excited by twilight UV) has decayed by then
tbl['MORNING'] = tbl['SUNSETD'] > 5

# =============================================================================
# Hot Star Filtering
# =============================================================================
# Hot stars (spectral type B/A) have featureless continua in the near-IR,
# making them ideal for telluric absorption measurements. Non-hot stars
# have intrinsic spectral features that can bias telluric fits.

# Get valid hot stars list from telluric config
telluric_config = load_telluric_config()
valid_hot_stars = telluric_config.get('hot_stars', [])

# Mark observations of confirmed hot stars
tbl['HOTSTAR'] = np.array([obj in valid_hot_stars for obj in tbl['DRSOBJN']])

n_total = len(tbl)
n_hotstar = np.sum(tbl['HOTSTAR'])
tprint(f'Initial sample: {n_total} spectra, {n_hotstar} are hot stars', color='green')

not_hot_stars = np.unique(tbl[tbl['HOTSTAR'] == False]['DRSOBJN'])

for obj in not_hot_stars:
    n_obj = np.sum(tbl['DRSOBJN'] == obj)
    tprint(f'  Excluding {obj}: {n_obj} spectra (not in hot_stars list)', color='blue')
    bad = (tbl['DRSOBJN'] == obj)
    tbl = tbl[~bad]

tprint(f'After hot star filter: {len(tbl)} spectra remaining', color='green')

    
# for all columns, try to convert to float, otherwise it's a string
for key in keys:

    if tbl[key][0] in [True,False]:
        tbl[key] = tbl[key].astype(bool)
        continue

    try:
        tbl[key] = tbl[key].astype(float)
    except:
        pass


# =============================================================================
# Quality Control: O2 Fluorescence Filtering
# =============================================================================
# O2 Meinel bands (emission) are excited by solar UV during twilight.
# This fluorescence contaminates O2 absorption measurements, causing O2_AIRM
# to deviate from the expected ~1.0 × airmass relationship.
# By midnight, UV-excited fluorescence has decayed (lifetime ~hours).
# We exclude observations with O2_AIRM outside the expected range.

# Load O2_AIRM acceptance thresholds from config
config = load_telluric_config()
qc_config = config.get('quality_control', {})
o2_airm_min = qc_config.get('o2_airm_min', 0.92)  # Lower bound (allows some noise)
o2_airm_max = qc_config.get('o2_airm_max', 1.08)  # Upper bound (catches fluorescence)

# Flag observations with valid O2 measurements (non-fluorescence-contaminated)
# Note: This ONLY affects O2 fitting. CO2/CH4/H2O use all data.
tbl['O2_VALID'] = (tbl['O2_AIRM'] >= o2_airm_min) & (tbl['O2_AIRM'] <= o2_airm_max)
n_o2_valid = np.sum(tbl['O2_VALID'])
n_o2_excluded = len(tbl) - n_o2_valid
tprint(f'O2_AIRM validity: {n_o2_valid} valid, {n_o2_excluded} excluded (range: {o2_airm_min}-{o2_airm_max})', color='blue')

# =============================================================================
# Data Quality Cuts
# =============================================================================
# Remove high-airmass observations (airmass > 2.0 corresponds to zenith angle > 60°)
# At high airmass: increased atmospheric variability, refraction effects, longer paths
bad = tbl['AIRMASS'] > 2.0
n_high_airmass = np.sum(bad)
tbl = tbl[~bad]
tprint(f'Removed {n_high_airmass} spectra with airmass > 2.0 ({len(tbl)} remaining)', color='blue')

# Remove low-SNR observations (unreliable fits)
bad = tbl['SNR_REF'] < 100
n_low_snr = np.sum(bad)
tbl = tbl[~bad]
tprint(f'Removed {n_low_snr} spectra with SNR < 100 ({len(tbl)} remaining)', color='blue')


# Initialize sigma for iterative outlier rejection loop
sigmax = np.inf

# Re-flag morning/evening after quality cuts
tbl['MORNING'] = tbl['SUNSETD'] > 5  # >5 hours after sunset (past midnight)
tbl['EVENING'] = tbl['SUNSETD'] < 5  # <5 hours after sunset (before midnight)

# =============================================================================
# Derived Quantities for Fitting
# =============================================================================
# Normalize environmental parameters to reference values:
# - Temperature normalized to 273.15 K (0°C)
# - Pressure ratio: sea-level-equivalent pressure / actual pressure
# These normalized values are used in power-law fits to capture physical scaling
tbl['NORMALIZED_PRESSURE'] = tbl['NORMPRES'] / tbl['PRESSURE']
tbl['NORMALIZED_TEMPERAT'] = tbl['TEMPERAT'] / 273.15


# Two-panel plot: O2_AIRM vs AIRMASS (left) and histogram (right)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]}, sharey=True)

# Compute ylim based on data points only
all_o2_airm = tbl['O2_AIRM']
y_margin = 0.02 * (np.nanmax(all_o2_airm) - np.nanmin(all_o2_airm))
ylim = (np.nanmin(all_o2_airm) - y_margin, np.nanmax(all_o2_airm) + y_margin)

# Left panel: O2_AIRM vs AIRMASS
# Valid O2 points
valid_morning_hot = tbl['MORNING'] & tbl['HOTSTAR'] & tbl['O2_VALID']
valid_morning_other = tbl['MORNING'] & ~tbl['HOTSTAR'] & tbl['O2_VALID']
ax1.plot(tbl['AIRMASS'][valid_morning_hot], tbl['O2_AIRM'][valid_morning_hot], 'ro', label='Hot star (valid)', alpha=0.7)
ax1.plot(tbl['AIRMASS'][valid_morning_other], tbl['O2_AIRM'][valid_morning_other], 'bo', label='Non hot star (valid)', alpha=0.7)
# Plot excluded O2 points in grey
o2_excluded = tbl['MORNING'] & ~tbl['O2_VALID']
n_excluded = np.sum(o2_excluded)
if n_excluded > 0:
    ax1.plot(tbl['AIRMASS'][o2_excluded], tbl['O2_AIRM'][o2_excluded], 'o', 
             color='grey', alpha=0.3, label=f'Excluded ({n_excluded})')
# Draw threshold lines
ax1.axhline(o2_airm_min, color='grey', linestyle='--', alpha=0.5, label=f'Threshold [{o2_airm_min:.2f}-{o2_airm_max:.2f}]')
ax1.axhline(o2_airm_max, color='grey', linestyle='--', alpha=0.5)
ax1.set_xlabel('AIRMASS')
ax1.set_ylabel('O2_AIRM')
ax1.set_ylim(ylim)
ax1.legend(loc='best', fontsize=8)

# Right panel: Histogram of O2_AIRM
bins = np.linspace(ylim[0], ylim[1], 30)
# Plot excluded points as grey histogram
if n_excluded > 0:
    ax2.hist(tbl['O2_AIRM'][~tbl['O2_VALID']], bins=bins, color='grey', alpha=0.5, label='Excluded', orientation='horizontal')
# Plot valid points as colored histogram
ax2.hist(tbl['O2_AIRM'][tbl['O2_VALID']], bins=bins, color='blue', alpha=0.7, label='Valid', orientation='horizontal')
# Draw threshold lines
ax2.axhline(o2_airm_min, color='grey', linestyle='--', alpha=0.5)
ax2.axhline(o2_airm_max, color='grey', linestyle='--', alpha=0.5)
ax2.set_xlabel('Count')
ax2.legend(loc='best', fontsize=8)

plt.tight_layout()

# Paper Figure 7: QC diagnostics (O2 airmass, outlier flagging)
enabled, output_dir = get_paper_figures_config(instrument)
if enabled and not _paper_figure_done['fig7']:
    fig_path = os.path.join(output_dir, 'fig7_qc_diagnostics.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    tprint(f'Paper figure saved: {fig_path}', color='green')
    _paper_figure_done['fig7'] = True

plt.savefig('o2_airmass_hotstar_check.png')
plt.show()

# Keep only hot star observations for fitting
tbl = tbl[tbl['HOTSTAR']]

# =============================================================================
# O2 Airmass Fit (Simple Scaling Model)
# =============================================================================
# O2 fitting uses ONLY morning observations with valid O2_AIRM values:
# 1. Morning (>midnight): Meinel band fluorescence has decayed by then
#    (fluorescence is excited by twilight UV, lifetime ~hours)
# 2. O2_VALID: O2_AIRM within expected range (not contaminated by residual fluorescence)
#
# O2 shows mild seasonal dependence (not expected, but observed).
# Use the full anthropic model like CO2/CH4 to capture temporal variations.
o2_fit_mask = tbl['MORNING'] & tbl['O2_VALID']

# Iterative sigma-clipping fit using anthropic model
while sigmax > sigma_cut:
    mjds_airmass_o2 = [tbl['MJDMID'][o2_fit_mask], tbl['AIRMASS'][o2_fit_mask], 
                       tbl['NORMALIZED_TEMPERAT'][o2_fit_mask], tbl['NORMALIZED_PRESSURE'][o2_fit_mask]]
    
    # Initial guesses: slope~0, intercept~1 (O2_AIRM normalized), small amplitude, phase=0, exponents~1
    curve_fit_o2, curve_fit_cov = curve_fit(anthropic, mjds_airmass_o2, tbl['O2_AIRM'][o2_fit_mask], 
                                             p0=[0.0, 1.0, 0.01, 0.0, 1.0, 1.0, 1.0])
    
    # Create "zenith" version of parameters for computing residuals
    curve_fit_o2_zenith = curve_fit_o2.copy()
    curve_fit_o2_zenith[4] = 0.0  # airmass exponent → no airmass dependence
    curve_fit_o2_zenith[5] = 0.0  # temperature exponent → reference temperature
    curve_fit_o2_zenith[6] = 0.0  # pressure exponent → reference pressure

    residual = tbl['O2_AIRM'][o2_fit_mask]/tbl['AIRMASS'][o2_fit_mask]**curve_fit_o2[4]/(tbl['NORMALIZED_TEMPERAT'][o2_fit_mask])**curve_fit_o2[5]/tbl['NORMALIZED_PRESSURE'][o2_fit_mask]**curve_fit_o2[6] - anthropic([tbl['MJDMID'][o2_fit_mask], np.ones(np.sum(o2_fit_mask)), 273.15*np.ones(np.sum(o2_fit_mask)), np.ones(np.sum(o2_fit_mask))], *curve_fit_o2_zenith)
    
    sigmax = np.max(np.abs(residual/mp.robust_nanstd(residual)))
    if sigmax > sigma_cut:
        tprint(f'Removing O2 outlier with {sigmax:.2f} sigma max deviation', color='yellow')
        maxid = np.argmax(np.abs(residual/mp.robust_nanstd(residual)))
        # Mark as invalid in O2_VALID instead of removing the row
        o2_fit_indices = np.where(o2_fit_mask)[0]
        tbl['O2_VALID'][o2_fit_indices[maxid]] = False
        o2_fit_mask = tbl['MORNING'] & tbl['O2_VALID']

# Extract uncertainties from covariance matrix
curve_fit_o2_cov = curve_fit_cov.copy()
curve_fit_o2_err = np.sqrt(np.diag(curve_fit_o2_cov))

# Store O2 parameters (now with temporal model like CO2/CH4)
tbl_params_fit['O2_SLOPE'] = curve_fit_o2[0]
tbl_params_fit['O2_INTERCEPT'] = curve_fit_o2[1]
tbl_params_fit['O2_AMP'] = curve_fit_o2[2]
tbl_params_fit['O2_PHASE'] = curve_fit_o2[3]
tbl_params_fit['O2_AIRMASS_EXP'] = curve_fit_o2[4]
tbl_params_fit['O2_TEMPERATURE_EXP'] = curve_fit_o2[5]
tbl_params_fit['O2_PRESSURE_EXP'] = curve_fit_o2[6]
# Store uncertainties
tbl_params_fit['O2_SLOPE_ERR'] = curve_fit_o2_err[0]
tbl_params_fit['O2_INTERCEPT_ERR'] = curve_fit_o2_err[1]
tbl_params_fit['O2_AMP_ERR'] = curve_fit_o2_err[2]
tbl_params_fit['O2_PHASE_ERR'] = curve_fit_o2_err[3]
tbl_params_fit['O2_AIRMASS_EXP_ERR'] = curve_fit_o2_err[4]
tbl_params_fit['O2_TEMPERATURE_EXP_ERR'] = curve_fit_o2_err[5]
tbl_params_fit['O2_PRESSURE_EXP_ERR'] = curve_fit_o2_err[6]
                                                        
tprint('O2 fit parameters (morning):', color='blue')
tprint(f'Slope: {curve_fit_o2[0]*365.24*100:.4f} ± {curve_fit_o2_err[0]*365.24*100:.4f} %/year', color='blue')
tprint(f'Intercept: {curve_fit_o2[1]:.6f} ± {curve_fit_o2_err[1]:.6f}', color='blue')
tprint(f'Sinusoidal amplitude: {curve_fit_o2[2]*100:.4f} ± {curve_fit_o2_err[2]*100:.4f} %', color='blue')
tprint(f'Sinusoidal phase: {curve_fit_o2[3]:.3f} ± {curve_fit_o2_err[3]:.3f} rad', color='blue')
tprint(f'Airmass exponent: {curve_fit_o2[4]:.6f} ± {curve_fit_o2_err[4]:.6f}', color='blue')
tprint(f'Temperature exponent: {curve_fit_o2[5]:.6f} ± {curve_fit_o2_err[5]:.6f}', color='blue')
tprint(f'Pressure exponent: {curve_fit_o2[6]:.6f} ± {curve_fit_o2_err[6]:.6f}', color='blue')

# Compute full residuals for diagnostics
tbl['RESIDUAL_O2'] = np.nan
o2_normalized = tbl['O2_AIRM'][o2_fit_mask]/tbl['AIRMASS'][o2_fit_mask]**curve_fit_o2[4]/(tbl['NORMALIZED_TEMPERAT'][o2_fit_mask])**curve_fit_o2[5]/tbl['NORMALIZED_PRESSURE'][o2_fit_mask]**curve_fit_o2[6]
o2_model_at_data = anthropic([tbl['MJDMID'][o2_fit_mask], np.ones(np.sum(o2_fit_mask)), 273.15*np.ones(np.sum(o2_fit_mask)), np.ones(np.sum(o2_fit_mask))], *curve_fit_o2_zenith)
tbl['RESIDUAL_O2'][o2_fit_mask] = o2_normalized - o2_model_at_data

plt.figure()
plt.plot(tbl['NORMALIZED_TEMPERAT'][o2_fit_mask], tbl['RESIDUAL_O2'][o2_fit_mask], 'ro')
plt.xlabel('Temperature [K]')
plt.ylabel('O2 fit residuals (morning)')
plt.grid()
plt.show()

# same with pressure
plt.figure()
plt.plot(tbl['NORMALIZED_PRESSURE'][o2_fit_mask], tbl['RESIDUAL_O2'][o2_fit_mask], 'ro')
plt.xlabel('Pressure [kPa]')
plt.ylabel('O2 fit residuals (morning)')
plt.grid()
plt.show()

tprint(f'O2 fit residuals std (morning): {mp.robust_nanstd(tbl["RESIDUAL_O2"][o2_fit_mask])*100:.4f}%', color='blue')

# =============================================================================
# Paper Figure: O2 Temporal Fit
# =============================================================================
# Compute normalized data (corrected to zenith)
o2_data_normalized = tbl['O2_AIRM'][o2_fit_mask]/tbl['AIRMASS'][o2_fit_mask]**curve_fit_o2[4]/(tbl['NORMALIZED_TEMPERAT'][o2_fit_mask])**curve_fit_o2[5]/tbl['NORMALIZED_PRESSURE'][o2_fit_mask]**curve_fit_o2[6]

fig_o2, ax_o2 = plt.subplots(figsize=(10, 6))
# Plot data first to establish xlim
sc_o2 = ax_o2.scatter(tbl['MJDMID'][o2_fit_mask], o2_data_normalized, c=tbl['AIRMASS'][o2_fit_mask], 
                      cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5, s=50, zorder=1)

# Get xlim from data, then extend model beyond to ensure full coverage
xlim_o2 = ax_o2.get_xlim()
xlim_range_o2 = xlim_o2[1] - xlim_o2[0]
xlim_o2_extended = (xlim_o2[0] - 0.02*xlim_range_o2, xlim_o2[1] + 0.02*xlim_range_o2)
mjds_fit_o2 = np.linspace(xlim_o2_extended[0], xlim_o2_extended[1], 1000)

# Recompute model and envelope over the full xlim range
o2_fit_curve = anthropic([mjds_fit_o2, np.ones_like(mjds_fit_o2), np.ones_like(mjds_fit_o2), np.ones_like(mjds_fit_o2)], *curve_fit_o2_zenith)
o2_partials = np.zeros((len(mjds_fit_o2), 7))
o2_partials[:, 0] = mjds_fit_o2
o2_partials[:, 1] = 1.0
o2_partials[:, 2] = np.cos(2.0*np.pi*(mjds_fit_o2 % 365.24)/365.24 + curve_fit_o2[3])
o2_partials[:, 3] = -curve_fit_o2[2] * np.sin(2.0*np.pi*(mjds_fit_o2 % 365.24)/365.24 + curve_fit_o2[3])
o2_partials[:, 4:7] = 0.0
o2_model_var = np.sum(o2_partials @ curve_fit_o2_cov * o2_partials, axis=1)
o2_model_err = np.sqrt(o2_model_var)

# Plot 1-sigma envelope (in front of data points)
ax_o2.fill_between(mjds_fit_o2, o2_fit_curve - o2_model_err, o2_fit_curve + o2_model_err, 
                   color='red', alpha=0.4, zorder=2)
# Plot model with slope and amplitude in legend (convert to % units, use absolute amplitude)
o2_slope_yr = curve_fit_o2[0] * 365.24 * 100  # %/year
o2_slope_yr_err = curve_fit_o2_err[0] * 365.24 * 100
o2_amp_pct = np.abs(curve_fit_o2[2]) * 100  # Force positive amplitude
o2_amp_pct_err = curve_fit_o2_err[2] * 100
o2_label = f'Model: {o2_slope_yr:.3f}±{o2_slope_yr_err:.3f} %/yr, A={o2_amp_pct:.3f}±{o2_amp_pct_err:.3f} %'
ax_o2.plot(mjds_fit_o2, o2_fit_curve, 'r-', linewidth=2, label=o2_label, zorder=3)
cbar = plt.colorbar(sc_o2, ax=ax_o2)
cbar.set_label('Airmass')
ax_o2.set_xlabel('MJD', fontsize=12)
ax_o2.set_ylabel(r'O$_2$ optical depth/airmass at zenith', fontsize=12)
ax_o2.legend(loc='upper left', fontsize=10)
ax_o2.grid(True, alpha=0.3)
# Add secondary x-axis with years
def mjd_to_year_o2(mjd, pos):
    return f"{2000 + (mjd - 51544.5)/365.25:.1f}"
ax_o2_top = ax_o2.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
ax_o2_top.xaxis.set_major_formatter(FuncFormatter(mjd_to_year_o2))
ax_o2_top.set_xlabel('Year', fontsize=12)
plt.tight_layout()
# Force xlim to model range after tight_layout
ax_o2.set_xlim(mjds_fit_o2[0], mjds_fit_o2[-1])

# Save paper figure
enabled, output_dir = get_paper_figures_config(instrument)
if enabled and not _paper_figure_done['fig_o2']:
    fig_path = os.path.join(output_dir, 'fig_o2_temporal_fit.pdf')
    fig_o2.savefig(fig_path, dpi=300, bbox_inches='tight')
    tprint(f'Paper figure saved: {fig_path}', color='green')
    _paper_figure_done['fig_o2'] = True
plt.show()

plt.figure()
plt.scatter(tbl['AIRMASS'][tbl['EVENING']], tbl['O2_AIRM'][tbl['EVENING']], c=tbl['DRSSUNEL'][tbl['EVENING']], cmap='viridis', alpha=0.3)
plt.plot(tbl['AIRMASS'][o2_fit_mask], tbl['O2_AIRM'][o2_fit_mask], 'ro', label='Morning observations (valid O2)', alpha=0.5)
plt.colorbar(label='Sun angle below horizon [hours]')
plt.xlabel('AIRMASS')
plt.ylabel('O2_AIRM')
plt.show()

# =============================================================================
# CO2 Fit (Full Anthropic Model with Temporal Variation)
# =============================================================================
# CO2 shows clear temporal patterns:
# - Secular increase: ~2.5 ppm/year (fossil fuel emissions)
# - Annual cycle: ~6 ppm amplitude (Northern Hemisphere biosphere)
#   Peak in May (before growing season), minimum in September (after uptake)
# The fit also accounts for airmass, temperature, and pressure scaling.

nsigmax = np.inf

# Iterative sigma-clipping fit
while nsigmax > sigma_cut:
    # Prepare input data array
    mjds_airmass = [tbl['MJDMID'], tbl['AIRMASS'], tbl['NORMALIZED_TEMPERAT'], tbl['NORMALIZED_PRESSURE']]
    
    # Initial guesses: slope=0, intercept=400 ppm, amplitude=50 ppm, phase=0, all exponents=1
    curve_fit_co2, curve_fit_cov = curve_fit(anthropic, mjds_airmass, tbl['CO2_VMR'], 
                                              p0=[0.0, 400.0, 50.0, 0.0, 1.0, 1.0, 1.0])
    
    # Create "zenith" version of parameters for computing residuals
    # Set environmental exponents to 0 to get "standardized" CO2 at zenith, ref T/P
    curve_fit_co2_zenith = curve_fit_co2.copy()
    curve_fit_co2_zenith[4] = 0.0  # airmass exponent → no airmass dependence
    curve_fit_co2_zenith[5] = 0.0  # temperature exponent → reference temperature
    curve_fit_co2_zenith[6] = 0.0  # pressure exponent → reference pressure

    tbl['RESIDUAL_CO2'] = tbl['CO2_VMR']/tbl['AIRMASS']**curve_fit_co2[4]/(tbl['NORMALIZED_TEMPERAT'])**curve_fit_co2[5]/tbl['NORMALIZED_PRESSURE']**curve_fit_co2[6] - anthropic([tbl['MJDMID'], np.ones_like(tbl['MJDMID']), 273.15*np.ones_like(tbl['MJDMID']), np.ones_like(tbl['MJDMID'])], *curve_fit_co2_zenith)

    nsigmax = np.max(np.abs(tbl['RESIDUAL_CO2']/mp.robust_nanstd(tbl['RESIDUAL_CO2'])))
    if nsigmax > sigma_cut:
        print(f'Removing outlier with {nsigmax:.2f} sigma max deviation')
        maxid = np.argmax(np.abs(tbl['RESIDUAL_CO2']/mp.robust_nanstd(tbl['RESIDUAL_CO2'])))
        tbl.remove_row(maxid)


# Extract uncertainties from covariance matrix
# Store CO2 covariance under named variable
curve_fit_co2_cov = curve_fit_cov.copy()
curve_fit_co2_err = np.sqrt(np.diag(curve_fit_co2_cov))

tbl_params_fit['CO2_SLOPE'] = curve_fit_co2[0]
tbl_params_fit['CO2_INTERCEPT'] = curve_fit_co2[1]
tbl_params_fit['CO2_AMP'] = curve_fit_co2[2]
tbl_params_fit['CO2_PHASE'] = curve_fit_co2[3]
tbl_params_fit['CO2_AIRMASS_EXP'] = curve_fit_co2[4]
tbl_params_fit['CO2_TEMPERATURE_EXP'] = curve_fit_co2[5]
tbl_params_fit['CO2_PRESSURE_EXP'] = curve_fit_co2[6]
# Store uncertainties
tbl_params_fit['CO2_SLOPE_ERR'] = curve_fit_co2_err[0]
tbl_params_fit['CO2_INTERCEPT_ERR'] = curve_fit_co2_err[1]
tbl_params_fit['CO2_AMP_ERR'] = curve_fit_co2_err[2]
tbl_params_fit['CO2_PHASE_ERR'] = curve_fit_co2_err[3]
tbl_params_fit['CO2_AIRMASS_EXP_ERR'] = curve_fit_co2_err[4]
tbl_params_fit['CO2_TEMPERATURE_EXP_ERR'] = curve_fit_co2_err[5]
tbl_params_fit['CO2_PRESSURE_EXP_ERR'] = curve_fit_co2_err[6]

# print fit parameters with uncertainties
tprint('CO2 fit parameters:', color='blue')
tprint(f'Slope: {curve_fit_co2[0]*365.24:.2f} ± {curve_fit_co2_err[0]*365.24:.2f} ppm/year', color='blue')
tprint(f'Intercept: {curve_fit_co2[1]:.2f} ± {curve_fit_co2_err[1]:.2f} ppm', color='blue')
tprint(f'Sinusoidal amplitude: {curve_fit_co2[2]:.2f} ± {curve_fit_co2_err[2]:.2f} ppm', color='blue')
tprint(f'Sinusoidal phase: {curve_fit_co2[3]:.3f} ± {curve_fit_co2_err[3]:.3f} rad', color='blue')
tprint(f'Airmass exponent: {curve_fit_co2[4]:.4f} ± {curve_fit_co2_err[4]:.4f}', color='blue')
tprint(f'Temperature exponent: {curve_fit_co2[5]:.4f} ± {curve_fit_co2_err[5]:.4f}', color='blue')
tprint(f'Pressure exponent: {curve_fit_co2[6]:.4f} ± {curve_fit_co2_err[6]:.4f}', color='blue')

# =============================================================================
# Paper Figure: CO2 Temporal Fit
# =============================================================================
# Compute normalized data (corrected to zenith)
co2_data_normalized = tbl['CO2_VMR']/tbl['AIRMASS']**curve_fit_co2[4]/(tbl['NORMALIZED_TEMPERAT'])**curve_fit_co2[5]/tbl['NORMALIZED_PRESSURE']**curve_fit_co2[6]

fig_co2, ax_co2 = plt.subplots(figsize=(10, 6))
# Plot data first to establish xlim
sc = ax_co2.scatter(tbl['MJDMID'], co2_data_normalized, c=tbl['AIRMASS'], 
                    cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5, s=50, zorder=1)

# Get xlim from data, then extend model beyond to ensure full coverage
xlim_co2 = ax_co2.get_xlim()
xlim_range_co2 = xlim_co2[1] - xlim_co2[0]
xlim_co2_extended = (xlim_co2[0] - 0.02*xlim_range_co2, xlim_co2[1] + 0.02*xlim_range_co2)
mjds_fit = np.linspace(xlim_co2_extended[0], xlim_co2_extended[1], 1000)

# Recompute model and envelope over the full xlim range
co2_fit = anthropic([mjds_fit, np.ones_like(mjds_fit), np.ones_like(mjds_fit), np.ones_like(mjds_fit)], *curve_fit_co2_zenith)
co2_partials = np.zeros((len(mjds_fit), 7))
co2_partials[:, 0] = mjds_fit
co2_partials[:, 1] = 1.0
co2_partials[:, 2] = np.cos(2.0*np.pi*(mjds_fit % 365.24)/365.24 + curve_fit_co2[3])
co2_partials[:, 3] = -curve_fit_co2[2] * np.sin(2.0*np.pi*(mjds_fit % 365.24)/365.24 + curve_fit_co2[3])
co2_partials[:, 4:7] = 0.0
co2_model_var = np.sum(co2_partials @ curve_fit_co2_cov * co2_partials, axis=1)
co2_model_err = np.sqrt(co2_model_var)

# Plot 1-sigma envelope (in front of data points)
ax_co2.fill_between(mjds_fit, co2_fit - co2_model_err, co2_fit + co2_model_err, 
                    color='red', alpha=0.4, zorder=2)
# Plot model with slope and amplitude in legend (use absolute amplitude)
co2_slope_yr = curve_fit_co2[0] * 365.24
co2_slope_yr_err = curve_fit_co2_err[0] * 365.24
co2_amp_abs = np.abs(curve_fit_co2[2])
co2_label = f'Model: {co2_slope_yr:.1f}±{co2_slope_yr_err:.1f} ppm/yr, A={co2_amp_abs:.1f}±{curve_fit_co2_err[2]:.1f} ppm'
ax_co2.plot(mjds_fit, co2_fit, 'r-', linewidth=2, label=co2_label, zorder=3)
cbar = plt.colorbar(sc, ax=ax_co2)
cbar.set_label('Airmass')
ax_co2.set_xlabel('MJD', fontsize=12)
ax_co2.set_ylabel(r'CO$_2$ VMR at zenith [ppm]', fontsize=12)
ax_co2.legend(loc='upper left', fontsize=10)
ax_co2.grid(True, alpha=0.3)
# Add secondary x-axis with years
def mjd_to_year(mjd, pos):
    return f"{2000 + (mjd - 51544.5)/365.25:.1f}"
ax_co2_top = ax_co2.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
ax_co2_top.xaxis.set_major_formatter(FuncFormatter(mjd_to_year))
ax_co2_top.set_xlabel('Year', fontsize=12)
plt.tight_layout()
# Force xlim to model range after tight_layout
ax_co2.set_xlim(mjds_fit[0], mjds_fit[-1])

# Save paper figure
enabled, output_dir = get_paper_figures_config(instrument)
if enabled and not _paper_figure_done['fig_co2']:
    fig_path = os.path.join(output_dir, 'fig_co2_temporal_fit.pdf')
    fig_co2.savefig(fig_path, dpi=300, bbox_inches='tight')
    tprint(f'Paper figure saved: {fig_path}', color='green')
    _paper_figure_done['fig_co2'] = True
plt.show()

tprint(f'Fractional CO2 fit residuals : {mp.robust_nanstd(tbl["RESIDUAL_CO2"])/np.nanmedian(tbl["CO2_VMR"])*100:.2f}%', color='blue')

# =============================================================================
# CH4 Fit (Full Anthropic Model with Temporal Variation)
# =============================================================================
# CH4 (methane) also shows temporal patterns:
# - Secular increase: ~10 ppb/year (wetlands, agriculture, fossil fuels)
# - Seasonal cycle: ~30-50 ppb amplitude (wetland emissions, OH sink variation)
# CH4 mixing ratio is ~1900 ppb (1.9 ppm), much lower than CO2 (~420 ppm)

nsigmax = np.inf

while nsigmax > sigma_cut:
    mjds_airmass = [tbl['MJDMID'], tbl['AIRMASS'], tbl['NORMALIZED_TEMPERAT'], tbl['NORMALIZED_PRESSURE']]
    
    # Initial guesses: intercept ~1500 ppb (older data), amplitude ~200 ppb
    curve_fit_ch4, curve_fit_cov = curve_fit(anthropic, mjds_airmass, tbl['CH4_VMR'], 
                                              p0=[0.0, 1500.0, 200.0, 0.0, 1.0, 1.0, 1.0])

    # Zenith-normalized version for residual calculation
    curve_fit_ch4_zenith = curve_fit_ch4.copy()
    curve_fit_ch4_zenith[4] = 0.0  # airmass exponent → no airmass dependence
    curve_fit_ch4_zenith[5] = 0.0  # temperature exponent → reference temperature
    curve_fit_ch4_zenith[6] = 0.0  # pressure exponent → reference pressure


    tbl['RESIDUAL_CH4'] = tbl['CH4_VMR']/tbl['AIRMASS']**curve_fit_ch4[4]/(tbl['NORMALIZED_TEMPERAT'])**curve_fit_ch4[5]/tbl['NORMALIZED_PRESSURE']**curve_fit_ch4[6] - anthropic([tbl['MJDMID'], np.ones_like(tbl['MJDMID']), 273.15*np.ones_like(tbl['MJDMID']), np.ones_like(tbl['MJDMID'])], *curve_fit_ch4_zenith)

    nsigmax = np.max(np.abs(tbl['RESIDUAL_CH4']/mp.robust_nanstd(tbl['RESIDUAL_CH4'])))
    if nsigmax > sigma_cut:
        print(f'Removing outlier with {nsigmax:.2f} sigma max deviation')
        maxid = np.argmax(np.abs(tbl['RESIDUAL_CH4']/mp.robust_nanstd(tbl['RESIDUAL_CH4'])))
        tbl.remove_row(maxid)


# Extend range by 5% margin so model extends to plot edges
mjd_min, mjd_max = np.min(tbl['MJDMID']), np.max(tbl['MJDMID'])
mjd_margin = 0.05 * (mjd_max - mjd_min)
mjds_fit = np.linspace(mjd_min - mjd_margin, mjd_max + mjd_margin, 1000)

# Extract uncertainties from covariance matrix
# Store CH4 covariance under named variable
curve_fit_ch4_cov = curve_fit_cov.copy()
curve_fit_ch4_err = np.sqrt(np.diag(curve_fit_ch4_cov))

tprint('CH4 fit parameters:', color='blue')
tprint(f'Slope: {curve_fit_ch4[0]*365.24:.4f} ± {curve_fit_ch4_err[0]*365.24:.4f} ppm/year', color='blue')
tprint(f'Intercept: {curve_fit_ch4[1]:.2f} ± {curve_fit_ch4_err[1]:.2f} ppm', color='blue')
tprint(f'Sinusoidal amplitude: {curve_fit_ch4[2]:.2f} ± {curve_fit_ch4_err[2]:.2f} ppm', color='blue')
tprint(f'Sinusoidal phase: {curve_fit_ch4[3]:.3f} ± {curve_fit_ch4_err[3]:.3f} rad', color='blue')
tprint(f'Airmass exponent: {curve_fit_ch4[4]:.4f} ± {curve_fit_ch4_err[4]:.4f}', color='blue')
tprint(f'Temperature exponent: {curve_fit_ch4[5]:.4f} ± {curve_fit_ch4_err[5]:.4f}', color='blue')
tprint(f'Pressure exponent: {curve_fit_ch4[6]:.4f} ± {curve_fit_ch4_err[6]:.4f}', color='blue')

tbl_params_fit['CH4_SLOPE'] = curve_fit_ch4[0]
tbl_params_fit['CH4_INTERCEPT'] = curve_fit_ch4[1]
tbl_params_fit['CH4_AMP'] = curve_fit_ch4[2]
tbl_params_fit['CH4_PHASE'] = curve_fit_ch4[3]
tbl_params_fit['CH4_AIRMASS_EXP'] = curve_fit_ch4[4]
tbl_params_fit['CH4_TEMPERATURE_EXP'] = curve_fit_ch4[5]
tbl_params_fit['CH4_PRESSURE_EXP'] = curve_fit_ch4[6]
# Store uncertainties
tbl_params_fit['CH4_SLOPE_ERR'] = curve_fit_ch4_err[0]
tbl_params_fit['CH4_INTERCEPT_ERR'] = curve_fit_ch4_err[1]
tbl_params_fit['CH4_AMP_ERR'] = curve_fit_ch4_err[2]
tbl_params_fit['CH4_PHASE_ERR'] = curve_fit_ch4_err[3]
tbl_params_fit['CH4_AIRMASS_EXP_ERR'] = curve_fit_ch4_err[4]
tbl_params_fit['CH4_TEMPERATURE_EXP_ERR'] = curve_fit_ch4_err[5]
tbl_params_fit['CH4_PRESSURE_EXP_ERR'] = curve_fit_ch4_err[6]

# =============================================================================
# Paper Figure: CH4 Temporal Fit
# =============================================================================
# Compute normalized data (corrected to zenith)
ch4_data_normalized = tbl['CH4_VMR']/tbl['AIRMASS']**curve_fit_ch4[4]/(tbl['NORMALIZED_TEMPERAT'])**curve_fit_ch4[5]/tbl['NORMALIZED_PRESSURE']**curve_fit_ch4[6]

fig_ch4, ax_ch4 = plt.subplots(figsize=(10, 6))
# Plot data first to establish xlim
sc = ax_ch4.scatter(tbl['MJDMID'], ch4_data_normalized, c=tbl['AIRMASS'], 
                    cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5, s=50, zorder=1)

# Get xlim from data, then extend model beyond to ensure full coverage
xlim_ch4 = ax_ch4.get_xlim()
xlim_range_ch4 = xlim_ch4[1] - xlim_ch4[0]
xlim_ch4_extended = (xlim_ch4[0] - 0.02*xlim_range_ch4, xlim_ch4[1] + 0.02*xlim_range_ch4)
mjds_fit_ch4 = np.linspace(xlim_ch4_extended[0], xlim_ch4_extended[1], 1000)

# Recompute model and envelope over the full xlim range
ch4_fit = anthropic([mjds_fit_ch4, np.ones_like(mjds_fit_ch4), np.ones_like(mjds_fit_ch4), np.ones_like(mjds_fit_ch4)], *curve_fit_ch4_zenith)
ch4_partials = np.zeros((len(mjds_fit_ch4), 7))
ch4_partials[:, 0] = mjds_fit_ch4
ch4_partials[:, 1] = 1.0
ch4_partials[:, 2] = np.cos(2.0*np.pi*(mjds_fit_ch4 % 365.24)/365.24 + curve_fit_ch4[3])
ch4_partials[:, 3] = -curve_fit_ch4[2] * np.sin(2.0*np.pi*(mjds_fit_ch4 % 365.24)/365.24 + curve_fit_ch4[3])
ch4_partials[:, 4:7] = 0.0
ch4_model_var = np.sum(ch4_partials @ curve_fit_ch4_cov * ch4_partials, axis=1)
ch4_model_err = np.sqrt(ch4_model_var)

# Plot 1-sigma envelope (in front of data points)
ax_ch4.fill_between(mjds_fit_ch4, ch4_fit - ch4_model_err, ch4_fit + ch4_model_err, 
                    color='red', alpha=0.4, zorder=2)
# Plot model with slope and amplitude in legend (use absolute amplitude)
ch4_slope_yr = curve_fit_ch4[0] * 365.24
ch4_slope_yr_err = curve_fit_ch4_err[0] * 365.24
ch4_amp_abs = np.abs(curve_fit_ch4[2])
ch4_label = f'Model: {ch4_slope_yr:.2f}±{ch4_slope_yr_err:.2f} ppm/yr, A={ch4_amp_abs:.1f}±{curve_fit_ch4_err[2]:.1f} ppm'
ax_ch4.plot(mjds_fit_ch4, ch4_fit, 'r-', linewidth=2, label=ch4_label, zorder=3)
cbar = plt.colorbar(sc, ax=ax_ch4)
cbar.set_label('Airmass')
ax_ch4.set_xlabel('MJD', fontsize=12)
ax_ch4.set_ylabel(r'CH$_4$ VMR at zenith [ppb]', fontsize=12)
ax_ch4.legend(loc='upper left', fontsize=10)
ax_ch4.grid(True, alpha=0.3)
# Add secondary x-axis with years
ax_ch4_top = ax_ch4.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
ax_ch4_top.xaxis.set_major_formatter(FuncFormatter(mjd_to_year))
ax_ch4_top.set_xlabel('Year', fontsize=12)
plt.tight_layout()
# Force xlim to model range after tight_layout
ax_ch4.set_xlim(mjds_fit_ch4[0], mjds_fit_ch4[-1])

# Save paper figure
enabled, output_dir = get_paper_figures_config(instrument)
if enabled and not _paper_figure_done['fig_ch4']:
    fig_path = os.path.join(output_dir, 'fig_ch4_temporal_fit.pdf')
    fig_ch4.savefig(fig_path, dpi=300, bbox_inches='tight')
    tprint(f'Paper figure saved: {fig_path}', color='green')
    _paper_figure_done['fig_ch4'] = True
plt.show()

tprint(f'Fractional CH4 fit residuals : {mp.robust_nanstd(tbl["RESIDUAL_CH4"])/np.nanmedian(tbl["CH4_VMR"])*100:.2f}%', color='blue')

plt.figure()
plt.plot(tbl['AIRMASS'],tbl['RESIDUAL_CH4'], 'ko', label='CH4 residuals')
plt.plot(tbl['AIRMASS'],tbl['RESIDUAL_CO2'], 'ro', label='CO2 residuals')
plt.xlabel('AIRMASS')
plt.ylabel('Residuals [ppm]')
plt.legend()
plt.show()

plt.figure()
plt.plot(tbl['PRESSURE'],tbl['RESIDUAL_CH4'], 'ko', label='CH4 residuals')
plt.plot(tbl['PRESSURE'],tbl['RESIDUAL_CO2'], 'ro', label='CO2 residuals')
plt.xlabel('PRESSURE')
plt.ylabel('Residuals [ppm]')
plt.legend()
plt.show()


tbl_params = Table()
key = np.array([key for key in tbl_params_fit.keys()])
tbl_params['PARAM'] = key
tbl_params['VALUE'] = np.array([tbl_params_fit[key] for key in tbl_params_fit.keys()])

# Save fitted parameters to CSV file for use by predict_abso.py
outpath = os.path.join(project_path, outname)
tbl_params.write(outpath, format='csv', overwrite=True)
tprint(f'Saved parameters to {outpath}', color='blue')

# =============================================================================
# Generate LaTeX Table for Paper
# =============================================================================
# Create a LaTeX-formatted file with all fitted parameters and uncertainties
# This can be directly included in the paper using \input{filename}

enabled, output_dir = get_paper_figures_config(instrument)
if enabled:
    latex_file = os.path.join(output_dir, f'telluric_fit_params_{instrument}.tex')
    
    with open(latex_file, 'w') as f:
        f.write('% Telluric absorption fit parameters for ' + instrument + '\\n')
        f.write('% Generated by compil_stats.py\\n')
        f.write('% Date: ' + str(np.datetime64('today')) + '\\n')
        f.write('% Number of hot star spectra used: ' + str(len(tbl)) + '\\n')
        f.write('\\n')
        
        # O2 parameters table (now with temporal model like CO2/CH4)
        f.write('% ===== O2 Temporal Model Parameters =====\\n')
        f.write('\\\\begin{table}[h]\\n')
        f.write('\\\\centering\\n')
        f.write('\\\\caption{O$_2$ temporal model parameters.}\\n')
        f.write('\\\\label{tab:o2_params}\\n')
        f.write('\\\\begin{tabular}{lcc}\\n')
        f.write('\\\\hline\\n')
        f.write('Parameter & Value & Unit \\\\\\\\\\n')
        f.write('\\\\hline\\n')
        o2_slope_pct_yr = tbl_params_fit["O2_SLOPE"] * 365.24 * 100  # Convert to %/yr
        o2_slope_err_pct_yr = tbl_params_fit["O2_SLOPE_ERR"] * 365.24 * 100
        f.write(f'Secular trend ($s$) & ${o2_slope_pct_yr:.4f} \\\\pm {o2_slope_err_pct_yr:.4f}$ & \\%/yr \\\\\\\\\\n')
        f.write(f'Base value ($c_0$) & ${tbl_params_fit["O2_INTERCEPT"]:.6f} \\\\pm {tbl_params_fit["O2_INTERCEPT_ERR"]:.6f}$ & -- \\\\\\\\\\n')
        o2_amp_pct = tbl_params_fit["O2_AMP"] * 100
        o2_amp_err_pct = tbl_params_fit["O2_AMP_ERR"] * 100
        f.write(f'Seasonal amplitude ($A$) & ${o2_amp_pct:.4f} \\\\pm {o2_amp_err_pct:.4f}$ & \\% \\\\\\\\\\n')
        f.write(f'Seasonal phase ($\\\\phi$) & ${tbl_params_fit["O2_PHASE"]:.3f} \\\\pm {tbl_params_fit["O2_PHASE_ERR"]:.3f}$ & rad \\\\\\\\\\n')
        f.write(f'Airmass exponent ($\\\\alpha$) & ${tbl_params_fit["O2_AIRMASS_EXP"]:.4f} \\\\pm {tbl_params_fit["O2_AIRMASS_EXP_ERR"]:.4f}$ & -- \\\\\\\\\\n')
        f.write(f'Temperature exponent ($\\\\beta$) & ${tbl_params_fit["O2_TEMPERATURE_EXP"]:.4f} \\\\pm {tbl_params_fit["O2_TEMPERATURE_EXP_ERR"]:.4f}$ & -- \\\\\\\\\\n')
        f.write(f'Pressure exponent ($\\\\gamma$) & ${tbl_params_fit["O2_PRESSURE_EXP"]:.4f} \\\\pm {tbl_params_fit["O2_PRESSURE_EXP_ERR"]:.4f}$ & -- \\\\\\\\\\n')
        f.write('\\\\hline\\n')
        f.write('\\\\end{tabular}\\n')
        f.write('\\\\end{table}\\n')
        f.write('\\n')
        
        # CO2 parameters table
        f.write('% ===== CO2 Temporal Model Parameters =====\\n')
        f.write('\\\\begin{table}[h]\\n')
        f.write('\\\\centering\\n')
        f.write('\\\\caption{CO$_2$ temporal model parameters: $\\\\mathrm{VMR}(t) = (s \\\\cdot t + c_0 + A \\\\cos(2\\\\pi t/\\\\mathrm{yr} + \\\\phi)) \\\\times \\\\mathrm{airmass}^\\\\alpha \\\\times T^\\\\beta \\\\times P^\\\\gamma$}\\n')
        f.write('\\\\label{tab:co2_params}\\n')
        f.write('\\\\begin{tabular}{lcc}\\n')
        f.write('\\\\hline\\n')
        f.write('Parameter & Value & Unit \\\\\\\\\\n')
        f.write('\\\\hline\\n')
        slope_ppm_yr = tbl_params_fit["CO2_SLOPE"] * 365.24
        slope_err_ppm_yr = tbl_params_fit["CO2_SLOPE_ERR"] * 365.24
        f.write(f'Secular trend ($s$) & ${slope_ppm_yr:.2f} \\\\pm {slope_err_ppm_yr:.2f}$ & ppm/yr \\\\\\\\\\n')
        f.write(f'Base concentration ($c_0$) & ${tbl_params_fit["CO2_INTERCEPT"]:.1f} \\\\pm {tbl_params_fit["CO2_INTERCEPT_ERR"]:.1f}$ & ppm \\\\\\\\\\n')
        f.write(f'Seasonal amplitude ($A$) & ${tbl_params_fit["CO2_AMP"]:.2f} \\\\pm {tbl_params_fit["CO2_AMP_ERR"]:.2f}$ & ppm \\\\\\\\\\n')
        f.write(f'Seasonal phase ($\\\\phi$) & ${tbl_params_fit["CO2_PHASE"]:.3f} \\\\pm {tbl_params_fit["CO2_PHASE_ERR"]:.3f}$ & rad \\\\\\\\\\n')
        f.write(f'Airmass exponent ($\\\\alpha$) & ${tbl_params_fit["CO2_AIRMASS_EXP"]:.4f} \\\\pm {tbl_params_fit["CO2_AIRMASS_EXP_ERR"]:.4f}$ & -- \\\\\\\\\\n')
        f.write(f'Temperature exponent ($\\\\beta$) & ${tbl_params_fit["CO2_TEMPERATURE_EXP"]:.4f} \\\\pm {tbl_params_fit["CO2_TEMPERATURE_EXP_ERR"]:.4f}$ & -- \\\\\\\\\\n')
        f.write(f'Pressure exponent ($\\\\gamma$) & ${tbl_params_fit["CO2_PRESSURE_EXP"]:.4f} \\\\pm {tbl_params_fit["CO2_PRESSURE_EXP_ERR"]:.4f}$ & -- \\\\\\\\\\n')
        f.write('\\\\hline\\n')
        f.write('\\\\end{tabular}\\n')
        f.write('\\\\end{table}\\n')
        f.write('\\n')
        
        # CH4 parameters table
        f.write('% ===== CH4 Temporal Model Parameters =====\\n')
        f.write('\\\\begin{table}[h]\\n')
        f.write('\\\\centering\\n')
        f.write('\\\\caption{CH$_4$ temporal model parameters.}\\n')
        f.write('\\\\label{tab:ch4_params}\\n')
        f.write('\\\\begin{tabular}{lcc}\\n')
        f.write('\\\\hline\\n')
        f.write('Parameter & Value & Unit \\\\\\\\\\n')
        f.write('\\\\hline\\n')
        ch4_slope_ppb_yr = tbl_params_fit["CH4_SLOPE"] * 365.24 * 1000  # Convert to ppb/yr
        ch4_slope_err_ppb_yr = tbl_params_fit["CH4_SLOPE_ERR"] * 365.24 * 1000
        f.write(f'Secular trend ($s$) & ${ch4_slope_ppb_yr:.2f} \\\\pm {ch4_slope_err_ppb_yr:.2f}$ & ppb/yr \\\\\\\\\\n')
        f.write(f'Base concentration ($c_0$) & ${tbl_params_fit["CH4_INTERCEPT"]:.1f} \\\\pm {tbl_params_fit["CH4_INTERCEPT_ERR"]:.1f}$ & ppb \\\\\\\\\\n')
        f.write(f'Seasonal amplitude ($A$) & ${tbl_params_fit["CH4_AMP"]:.1f} \\\\pm {tbl_params_fit["CH4_AMP_ERR"]:.1f}$ & ppb \\\\\\\\\\n')
        f.write(f'Seasonal phase ($\\\\phi$) & ${tbl_params_fit["CH4_PHASE"]:.3f} \\\\pm {tbl_params_fit["CH4_PHASE_ERR"]:.3f}$ & rad \\\\\\\\\\n')
        f.write(f'Airmass exponent ($\\\\alpha$) & ${tbl_params_fit["CH4_AIRMASS_EXP"]:.4f} \\\\pm {tbl_params_fit["CH4_AIRMASS_EXP_ERR"]:.4f}$ & -- \\\\\\\\\\n')
        f.write(f'Temperature exponent ($\\\\beta$) & ${tbl_params_fit["CH4_TEMPERATURE_EXP"]:.4f} \\\\pm {tbl_params_fit["CH4_TEMPERATURE_EXP_ERR"]:.4f}$ & -- \\\\\\\\\\n')
        f.write(f'Pressure exponent ($\\\\gamma$) & ${tbl_params_fit["CH4_PRESSURE_EXP"]:.4f} \\\\pm {tbl_params_fit["CH4_PRESSURE_EXP_ERR"]:.4f}$ & -- \\\\\\\\\\n')
        f.write('\\\\hline\\n')
        f.write('\\\\end{tabular}\\n')
        f.write('\\\\end{table}\\n')
        f.write('\\n')
        
        # Summary statistics
        f.write('% ===== Fit Quality Statistics =====\\n')
        co2_rms = mp.robust_nanstd(tbl["RESIDUAL_CO2"])/np.nanmedian(tbl["CO2_VMR"])*100
        ch4_rms = mp.robust_nanstd(tbl["RESIDUAL_CH4"])/np.nanmedian(tbl["CH4_VMR"])*100
        f.write(f'% CO2 fractional RMS residuals: {co2_rms:.2f}%\\n')
        f.write(f'% CH4 fractional RMS residuals: {ch4_rms:.2f}%\\n')
        f.write(f'% Number of observations: {len(tbl)}\\n')
        f.write(f'% MJD range: {np.min(tbl["MJDMID"]):.1f} - {np.max(tbl["MJDMID"]):.1f}\\n')
        
    tprint(f'LaTeX parameters table saved: {latex_file}', color='green')

# =============================================================================
# Compute Mean Zenith Exponents and Identify Main Absorbers
# =============================================================================
# The "exponent" for each molecule represents its optical depth at zenith.
# By dividing the fitted exponent by airmass and taking the median,
# we get the typical zenith optical depth for each molecule.

mean_expos = [
    np.nanmedian(tbl['EXPO_H2O'] / tbl['AIRMASS']),  # H2O: highly variable
    np.nanmedian(tbl['EXPO_CO2'] / tbl['AIRMASS']),  # CO2: ~420 ppm
    np.nanmedian(tbl['EXPO_CH4'] / tbl['AIRMASS']),  # CH4: ~1.9 ppm
    np.nanmedian(tbl['EXPO_O2'] / tbl['AIRMASS'])    # O2: 20.95%
]
tprint('Mean exponents at zenith:', color='green')
tprint(f'H2O: {mean_expos[0]:.3f}, CO2: {mean_expos[1]:.3f}, CH4: {mean_expos[2]:.3f}, O2: {mean_expos[3]:.3f}', color='blue')

hdr = fits.getheader(files[0])

# Construct absorption spectra for each molecule at mean exponent levels
# all_abso shape: (4, n_orders, n_pixels) for H2O, CO2, CH4, O2
all_abso = construct_abso(waveref, mean_expos, all_abso=None)
mean_abso = np.product(all_abso, axis=0)  # Combined transmission (product of all molecules)

# Define a "ceiling" absorption level (95% transmission)
# Wavelengths with all molecules > 95% are considered "clean"
ceil_abso = np.ones_like(all_abso[0]) * 0.95

# Append ceiling as 5th "pseudo-absorber" (index 4 = "none/clean")
all_abso = np.concatenate([all_abso, ceil_abso[None, :, :]], axis=0)

# Identify the dominant (minimum transmission = strongest) absorber at each wavelength
# This creates a map: 0=H2O, 1=CO2, 2=CH4, 3=O2, 4=none
main_absorber = np.argmin(all_abso, axis=0)

# Save main absorber map for telluric correction wavelength selection
outname = 'main_absorber_' + instrument + '.fits'
outpath = os.path.join(project_path, outname)
fits.writeto(outpath, main_absorber.astype(np.int16), overwrite=True)
tprint(f'Saved main absorber map to {outpath}', color='blue')

"""

# Define colors for each absorber
colors = ['blue', 'red', 'green', 'orange', 'gray']
labels = ['H2O', 'CH4', 'CO2', 'O2', 'none']

fig, ax = plt.subplots(figsize=(12, 6))

# Plot each order, color-coded by main absorber
for iord in range(mean_abso.shape[0]):
    for absorber in range(5):
        # Create copies with NaNs where absorber doesn't match
        wave_plot = waveref[iord].copy()
        abso_plot = mean_abso[iord].copy()
        
        # Keep current absorber AND its neighbors (transition points)
        mask = main_absorber[iord] == absorber
        # Extend mask by one point on each side
        mask_extended = mask.copy()
        mask_extended[1:] |= mask[:-1]   # include left neighbor
        mask_extended[:-1] |= mask[1:]   # include right neighbor
        
        wave_plot[~mask_extended] = np.nan
        abso_plot[~mask_extended] = np.nan
        
        ax.plot(wave_plot, abso_plot, color=colors[absorber], alpha=0.5, linewidth=1)

# Create legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=labels[i]) 
                   for i in range(5)]
ax.legend(handles=legend_elements)

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Mean absorption')
plt.show()

"""