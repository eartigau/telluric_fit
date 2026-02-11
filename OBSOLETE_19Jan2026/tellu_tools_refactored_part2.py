"""
Telluric Tools - Part 2: Velocity, Template, and Absorption Functions

This module contains the remaining functions from tellu_tools:
- Velocity determination
- Template fetching and handling
- Absorption modeling
- Header updates
- Airmass calculations

Import this as part of tellu_tools_refactored.
"""

import os
import numpy as np
import warnings
from typing import Dict, Tuple, Optional, List

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
import astropy.units as u

from scipy.optimize import curve_fit, minimize
from scipy.interpolate import InterpolatedUnivariateSpline as ius
import numexpr as ne
from tqdm import tqdm

from aperocore import math as mp
from aperocore.science import wavecore

from tellu_tools_config import (
    get_user_params,
    get_calib_paths,
    MOLECULES,
    SPEED_OF_LIGHT,
    VELOCITY_CONFIG,
    EXPONENT_OPT_CONFIG,
    TEMPLATE_CONFIG,
    AIRMASS_CONFIG,
    CONVOLUTION_CONFIG,
    HEADER_KEYS,
)


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

    print(f'Optimal velocity shift: {popt[1]:.2f} km/s')

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
    tapas_file = get_calib_paths(instrument)['tapas_file']
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
                  instrument: str = 'NIRPS') -> Tuple[ius, ius]:
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
        print("Using HIERARCH ESO TEL TARG RADVEL for systemic velocity")
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


# Export commonly used functions for convenience
__all__ = [
    'gauss',
    'get_velo',
    'update_header',
    'hotstar',
    'accurate_airmass',
    'delay_since_sunset',
    'weight_fall',
    'savgol_filter_nan_fast',
    'fetch_template',
]
