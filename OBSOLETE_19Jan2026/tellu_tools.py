from matplotlib.artist import get
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from tqdm import tqdm

import astropy.units as u
from scipy.optimize import curve_fit

import os
import warnings

import numexpr as ne
import numpy as np

import astropy.units as u

from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from aperocore import math as mp
from aperocore.science import wavecore

import matplotlib.pyplot as plt
# Speed of light in km/s (used in variable_res_conv)
speed_of_light = 299792.458

instrument = 'NIRPS'
molecules = ['H2O', 'CH4', 'CO2', 'O2']


def sky_pca_fast(wave=None, spectrum=None, sky_dict=None, force_positive=True, 
                 doplot=False, verbose=True):
    """
    Fit sky PCA components to a spectrum.
    
    Optimizations vs original version:
    - Analytical gradient for 10-100x faster convergence
    - Least squares initialization
    - Pre-flattened arrays to avoid repeated .ravel() calls
    
    Parameters
    ----------
    wave : array (N, M)
        Wavelength grid in nm
    spectrum : array (N, M)
        Spectrum to fit
    sky_dict : dict or None
        Dictionary with 'SCI_SKY' and 'WAVE'. If None, loads from files and returns dict.
    force_positive : bool
        If True, output sky is clipped to >= 0
    doplot : bool
        If True, displays a diagnostic plot
    verbose : bool
        If True, prints progress
        
    Returns
    -------
    sky_out : array (N, M)
        Fitted sky model, or sky_dict if sky_dict was None
    """
    if sky_dict is None:
        sky_file = os.path.join(user_params()['project_path'], 
                                f'sky_{instrument}/sky_pca_components.fits')
        waveref = os.path.join(user_params()['project_path'], 
                               f'calib_{instrument}/waveref.fits')
        sky_dict = {
            'SCI_SKY': fits.getdata(sky_file),
            'WAVE': fits.getdata(waveref)
        }
        return sky_dict
    
    Npca = sky_dict['SCI_SKY'].shape[0]
    
    # Interpolate PCA components onto the wavelength grid
    cube = np.zeros((Npca, *wave.shape))
    for ipca in range(Npca):
        cube[ipca] = wavecore.wave_to_wave(
            sky_dict['SCI_SKY'][ipca].reshape(wave.shape),
            sky_dict['WAVE'], wave
        )
    
    # Pre-flatten arrays (avoids repeated .ravel() calls)
    wave_flat = wave.ravel()
    spectrum_flat = spectrum.ravel().astype(np.float64)
    cube_flat = cube.reshape(Npca, -1).astype(np.float64)
    
    sky_out = np.zeros_like(spectrum_flat)
    
    # Spectral band definitions
    bands = [(950, 1400, 'Y+J'), (1400, 1900, 'H')]
    
    for wavemin, wavemax, band_name in bands:
        if verbose:
            print(f'\tFitting sky in {band_name} band ({wavemin}-{wavemax} nm)')
        
        # Spectral domain mask
        domain = (wave_flat > wavemin) & (wave_flat < wavemax)
        n_domain = np.sum(domain)
        
        # Extract domain data (contiguous arrays for speed)
        spec_dom = np.ascontiguousarray(spectrum_flat[domain])
        cube_dom = np.ascontiguousarray(cube_flat[:, domain])
        
        # Valid pixel mask (no NaN)
        valid_mask = np.isfinite(spec_dom) & np.all(np.isfinite(cube_dom), axis=0)
        n_valid = np.sum(valid_mask)

        # NaN-safe cube for gradient calculation (precomputed once)
        cube_dom_safe = np.where(np.isfinite(cube_dom), cube_dom, 0)
        
        if n_valid < Npca:
            if verbose:
                print(f'\t  Not enough valid pixels ({n_valid}), skipping')
            continue
        
        # Least squares initialization
        spec_valid = spec_dom[valid_mask]
        cube_valid = cube_dom[:, valid_mask]
        x0 = np.linalg.lstsq(cube_valid.T, spec_valid, rcond=None)[0]
        
        def compute_sky(amps):
            """Compute the sky model."""
            sky = np.dot(amps, cube_dom_safe)
            if force_positive:
                sky = np.maximum(sky, 0)
            return sky
        
        def objective_and_gradient(amps):
            """Robust objective and analytical gradient."""
            sky = compute_sky(amps)
            residual = spec_dom - sky
            
            # Robust RMS via MAD
            res_valid = residual[valid_mask]
            rms = np.nanmedian(np.abs(np.diff(res_valid))) + 1e-10
            
            # Robust weights
            nsig = res_valid / rms
            p_valid_prob = np.exp(-0.5 * nsig**2)
            weights = p_valid_prob / (p_valid_prob + 1e-4)
            
            # Weighted chi-square
            obj = np.sum(weights * res_valid**2) / n_valid
            
            # Analytical gradient
            weighted_res = np.zeros(n_domain)
            weighted_res[valid_mask] = weights * res_valid
            
            if force_positive:
                # Zero gradient where sky is clipped
                sky_unclipped = np.dot(amps, cube_dom_safe)
                weighted_res = weighted_res * (sky_unclipped >= 0)
            
            grad = -2.0 * np.dot(cube_dom_safe, weighted_res) / n_valid
            
            return obj, grad
        
        # L-BFGS-B optimization with analytical gradient
        result = minimize(
            objective_and_gradient,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-3}
        )
        
        if verbose:
            print(f'\t  Converged: {result.success}, iterations: {result.nit}, '
                  f'Chi2: {result.fun:.4f}')
        
        # Apply final model
        sky_out[domain] += compute_sky(result.x)
    
    if doplot:
        plt.figure(figsize=(12, 4))
        plt.plot(wave_flat, spectrum_flat-sky_out, 'g', alpha=0.8, lw=1, label='Sky residual')
        plt.plot(wave_flat, spectrum_flat, 'b', alpha=0.5, lw=0.5, label='Spectrum')
        plt.plot(wave_flat, sky_out, 'r', alpha=0.8, lw=0.5, label='Sky model')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return sky_out.reshape(spectrum.shape)


if instrument == 'NIRPS':
    wave_fit = [980,1800]  # wavelength fit range in nm
if instrument == 'SPIROU':
    wave_fit = [980,2400]  # wavelength fit range in nm


def user_params():
    # Unique paths to recognize the computers

    path = '/project/6102120'
    # this is alliance server
    if os.path.exists(path):
        param_dict = {'project_path': '/project/6102120/eartigau/tapas/test_fit/', 
                      'doplot' : False, 'knee' : 0.3, 'wave_fit': wave_fit}

    else:
        # local computer
        param_dict = {'project_path': '/Users/eartigau/test_fit/', 
                      'doplot' : False, 'knee' : 0.3, 'wave_fit': wave_fit}
        
    return param_dict

if instrument == 'NIRPS':
    E2DS_FWHM = fits.getdata(os.path.join(user_params()['project_path'],f'calib_{instrument}/C7A164F31A_pp_e2dsff_A_waveref_res_e2ds_A.fits'),'E2DS_FWHM')
    E2DS_EXPO = fits.getdata(os.path.join(user_params()['project_path'],f'calib_{instrument}/C7A164F31A_pp_e2dsff_A_waveref_res_e2ds_A.fits'),'E2DS_EXPO')
    blaze = fits.getdata(os.path.join(user_params()['project_path'],f'calib_{instrument}/07337C08CA_pp_blaze_A.fits'))

elif instrument == 'SPIROU':
    E2DS_FWHM = fits.getdata(os.path.join(user_params()['project_path'],f'calib_{instrument}/3444961B5Da_pp_e2dsff_AB_waveref_res_e2ds_AB.fits'),'E2DS_FWHM')
    E2DS_EXPO = fits.getdata(os.path.join(user_params()['project_path'],f'calib_{instrument}/3444961B5Da_pp_e2dsff_AB_waveref_res_e2ds_AB.fits'),'E2DS_EXPO')
    blaze = fits.getdata(os.path.join(user_params()['project_path'],f'calib_{instrument}/5ABA102B11f_pp_blaze_AB.fits'))

else:
    raise ValueError('Instrument not recognized')


for iord in range(blaze.shape[0]):
    blaze[iord]/=np.nanpercentile(blaze[iord],90)



transm_file = os.path.join(user_params()['project_path'],'LaSilla_tapas.fits')


def get_header_transm():
    return fits.getheader(transm_file)

def get_blaze():
    return blaze


def gauss(x, a, x0, sigma,zp,expo):
    return a * np.exp(-0.5 * np.abs((x - x0) / sigma) ** expo)+zp

def get_velo(wave, sp, spl,dv_amp = 200, doplot = True):

    # compute the velocity shift between spec and
    dvs = np.arange(-dv_amp,dv_amp+1,0.5,dtype = float)

    with np.errstate(invalid='ignore'):
        sp_tmp = np.log(sp).ravel()
    sp_tmp-=mp.lowpassfilter(sp_tmp,101)

    amp = np.zeros_like(dvs,dtype = float)+np.nan
    #amps = np.zeros_like(dvs,dtype = float)+np.nan
    rms = mp.robust_nanstd(np.diff(sp_tmp))

    norm=np.nansum(sp_tmp**2)
    for i in tqdm(range(len(dvs))[::10], desc = 'Optimizing velocity shift', leave=False):
        dv = dvs[i]
        template2 = np.log(spl(wave*mp.relativistic_waveshift(dv))).ravel()
        amp[i] = np.nansum(sp_tmp*template2)#/norm
        #amp[i], amps[i] = mp.odd_ratio_mean(sp_tmp/template2,rms/template2)

    v0 = dvs[np.nanargmax(amp)]
    for i in tqdm(range(len(dvs)), desc = 'Optimizing velocity shift', leave=False):
        if np.isfinite(amp[i]):
            continue
        if np.abs(dvs[i]-v0)>20:
            continue

        dv = dvs[i]
        template2 = np.log(spl(wave*mp.relativistic_waveshift(dv))).ravel()
        amp[i] = np.nansum(sp_tmp*template2)#/norm


    keep = np.isfinite(amp)
    dvs = dvs[keep]
    amp = amp[keep]

    p0 = [np.nanmax(amp), dvs[np.nanargmax(amp)], 5.0,0,2]
    popt, pcov = curve_fit(gauss, dvs, amp, p0=p0)

    print(f'Optimal velocity shift: {popt[1]:.2f} km/s')

    if doplot:
        plt.plot(dvs, amp, label='Amplitude')
        # have a 1-sigma envelope
        plt.plot(dvs, amp, color='gray', alpha=0.3)
        plt.plot(dvs,gauss(dvs, *popt), label='Gaussian fit', color='red')

        plt.xlabel('Velocity shift (m/s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title('Velocity shift optimization')
        plt.show()

    return popt[1]

def update_header(hdr):
    if instrument == 'NIRPS':
        airmass_key1 = 'ESO TEL AIRM START'
        airmass_key2 = 'ESO TEL AIRM END'
        airmass = (hdr[airmass_key1] + hdr[airmass_key2]) / 2.0
        hdr['AIRMASS'] = airmass,'Mean Airmass during observation'
    elif instrument == 'SPIROU':
        print('We already have the airmass')
        airmass = hdr['AIRMASS']
    else:
        raise ValueError('Instrument not recognized')
    

    hdr['ACCAIRM'] = accurate_airmass(hdr), 'Accurate Airmass from MJDMID'

    if instrument == 'NIRPS':
        pressure_key1 = 'ESO TEL AMBI PRES START'
        pressure_key2 = 'ESO TEL AMBI PRES END'
        pressure = (hdr[pressure_key1] + hdr[pressure_key2]) / 2.0
        hdr['PRESSURE'] = pressure, '[kPa] Ambient Pressure at telescope'
    if instrument == 'SPIROU':
        print('We already have the pressure')
        pressure = hdr['PRESSURE']
    
    if instrument == 'NIRPS':
        hdr['TEMPERAT'] = hdr['HIERARCH ESO TEL AMBI TEMP']+273.15,'Ambient Temperature [K]'
    if instrument == 'SPIROU':
        hdr['TEMPERAT'] = hdr['TEMPERAT']+273.15,'Ambient Temperature [K]'
    
    hdr['SUNSETD'] = delay_since_sunset(hdr['MJDMID'], hdr = hdr), 'Delay since last sunset [hours]'

    if instrument == 'NIRPS':
        hdr['HUMIDITY'] = hdr['HIERARCH ESO TEL AMBI RHUM'],'Relative Humidity [%]'
    if instrument == 'SPIROU':
        hdr['HUMIDITY'] = hdr['RELHUMID']

    if instrument == 'NIRPS':
        transm_file = '../LaSilla_tapas.fits'
    elif instrument == 'SPIROU':
        transm_file = '../MaunaKea_tapas.fits'
    else:  
        raise ValueError('Instrument not recognized')
    
    hdr_transm = fits.getheader(transm_file)
    pressure0 = hdr_transm['PAMBIENT']
    hdr['NORMPRES'] = pressure0,'[kPa] Normalization pressure for TAPAS values'

    if instrument == 'NIRPS':
        hdr['SNR_REF'] = hdr['EXTSN060'],'SNR on reference star at ~1.60 micron'
    if instrument == 'SPIROU':
        hdr['SNR_REF'] = hdr['EXTSN042'],'SNR on reference star at ~1.60 micron'

    hdr['H2OCV'] = hdr_transm['H2OCV'],'TAPAS H2O column value [cm]'
    hdr['VMR_CO2'] = hdr_transm['VMR_CO2'],'TAPAS CO2 volume mixing ratio [ppm]'
    hdr['VMR_CH4'] = hdr_transm['VMR_CH4'],'TAPAS CH4 volume mixing ratio [ppm]'


    return hdr

def getdata_safe(filename, ext=None):
    """
    Open a FITS file and return the first data-containing HDU (if ext is None),
    or the data of a specific extension.

    Returns a copy of the data to avoid side effects if the source HDU is closed.
    """
    with fits.open(filename) as hdulist:
        if ext is None:
            # Return the first HDU that actually contains data
            for hdu in hdulist:
                if hdu.data is not None:
                    return hdu.data.copy()
            raise ValueError(f"Aucune donnée trouvée dans {filename}")
        else:
            data = hdulist[ext].data
            if data is None:
                raise ValueError(f"Pas de données dans l'extension {ext}")
            return data.copy()
    # Unreachable; kept for completeness
    return data

def getheader_safe(filename, ext=0):
    """
    Open a FITS file and return a copy of the header for the requested extension.
    """
    with fits.open(filename) as hdulist:
        header = hdulist[ext].header.copy()
    return header

def savgol_filter_nan_fast(y, window_length, polyorder, deriv=0, frac_valid=user_params()['knee']):
    """
    Savitzky-Golay-like smoothing handling NaNs efficiently.

    - For each central pixel, fit a polynomial on the window using only non-NaN
      samples via a least-squares Vandermonde system.
    - If too few valid samples (controlled by frac_valid), return NaN at that point.
    - Supports derivative estimates at the window center.

    Parameters
    ----------
    y : array_like
        Data to be filtered (NaNs allowed)
    window_length : int
        Odd-length window
    polyorder : int
        Polynomial order
    deriv : int
        Derivative order at the center (0 = smoothed value)
    frac_valid : float
        Minimum fraction of valid samples in the window to accept the fit
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

        if fraction_valid < frac_valid:
            y_filtered[i] = np.nan
            continue

        if n_valid <= polyorder:
            y_filtered[i] = np.nan
            continue

        x_valid = np.arange(window_length)[valid_mask] - half_window
        A = np.vander(x_valid, polyorder + 1, increasing=True)
        y_valid = window_data[valid_mask]

        coeffs, _, _, _ = np.linalg.lstsq(A, y_valid, rcond=None)

        if deriv == 0:
            y_filtered[i] = coeffs[0]
        else:
            if deriv <= polyorder:
                factorial = np.math.factorial(deriv)
                y_filtered[i] = coeffs[deriv] * factorial
            else:
                y_filtered[i] = 0
    return y_filtered


def make_t(FluxA, WaveA, BlazeA, Recon, SKYCORR_SCI=None, SKYCORR_CAL=None,hdr=None):

    """
    0  PRIMARY       1 PrimaryHDU     646   ()      
    1  FluxA         1 ImageHDU      1509   (4088, 75)   float64   
    2  WaveA         1 ImageHDU      1231   (4088, 75)   float64   
    3  BlazeA        1 ImageHDU       550   (4088, 75)   float64   
    4  Recon         1 ImageHDU      1509   (4088, 75)   float64   
    5  SKYCORR_SCI    1 ImageHDU      1505   (4088, 75)   float64   
    6  SKYCORR_CAL    1 ImageHDU      1505   (4088, 75)   float64 
    """

    dict_t = dict(PRIMARY = None,FluxA=FluxA,WaveA=WaveA,BlazeA=BlazeA,Recon=Recon,SKYCORR_SCI=SKYCORR_SCI,SKYCORR_CAL=SKYCORR_CAL)
    t = fits.HDUList()()
    for key in dict_t.keys():
        if key == 'PRIMARY':
            hdu = fits.PrimaryHDU()
        else:
            hdu = fits.ImageHDU(dict_t[key],name=key)
        t.append(hdu)
    return t



def sky_pca(wave = None,spectrum = None,sky_dict = None, force_positive = True, doplot = False):
    if sky_dict is None:
        sky_file = os.path.join(user_params()['project_path'],f'sky_{instrument}/sky_pca_components.fits')
        waveref = os.path.join(user_params()['project_path'],f'calib_{instrument}/waveref.fits')
        sky_dict= dict()
        sky_dict['SCI_SKY'] = fits.getdata(sky_file)
        sky_dict['WAVE'] = fits.getdata(waveref)
        return sky_dict
    
    Npca = sky_dict['SCI_SKY'].shape[0]
    cube = np.zeros((Npca,*wave.shape))
    for ipca in range(Npca):
        cube[ipca] = wavecore.wave_to_wave(sky_dict['SCI_SKY'][ipca].reshape(wave.shape),sky_dict['WAVE'],wave)


    sky_out = np.zeros_like(spectrum).ravel()
    for Y_JH in range(2):
        
        # correct domain
        if Y_JH == 0:
            wavemin, wavemax = 950,1400
            print('\tWe fit sky in Y+J band')
        else:
            wavemin, wavemax = 1400,1900
            print('\tWe fit sky in H band')

        # find which pixels are >3sig away from the mean
        domain = (wave.ravel()>wavemin) & (wave.ravel()<wavemax)

        x0 = np.zeros(Npca)

        for iamp in range(Npca):
            x0[iamp] = np.nansum(spectrum.ravel()[domain]*cube[iamp].ravel()[domain]) / np.nansum(cube[iamp].ravel()[domain]**2)

        def apply_amps(amps):
            sky0 = np.zeros(cube[0].shape).ravel()
            for ipca in range(Npca):
                sky0[domain]+= cube[ipca].ravel()[domain]*amps[ipca]

            if force_positive:
                sky0[domain][sky0[domain]<0]=0.0
            return sky0

        print('\nFitting sky PCA components in band {:.0f}-{:.0f} nm\n'.format(wavemin,wavemax))
        def model_q(amps):
            diff = (spectrum.ravel() - apply_amps(amps))[domain]
            rms = np.nanmedian(np.abs(np.diff(diff)))
            nsig = diff/rms
            p_valid = np.exp(-0.5*nsig**2)
            p_invalid = 1e-4
            w = p_valid/(p_valid + p_invalid)

            d2 = np.nanmean(w*(diff)**2)

            print('\r Chi2 = {:.6f}      '.format(d2),end='',flush=True)

            return d2
        
        x = minimize(model_q, x0)

        #plt.plot(apply_amps(x.x))

        sky_out += apply_amps(x.x)
    #plt.show()

    sky_out = sky_out.reshape(spectrum.shape)

    if doplot:
        plt.plot(wave.ravel(), (spectrum-sky_out).ravel(), label='Original Spectrum', color='blue')
        plt.plot(wave.ravel(), sky_out.ravel(), label='Reconstructed Sky', color='red',alpha = 0.5)
        plt.show()

    return sky_out

def recon_sky(wave=None,spectrum=None, sky_dict = None, force_positive = True):
    """
    Docstring pour subtract_oh
    
    :param wave: Description
    :param spectrum: Description
    :param sky_dict: Description
    :param force_positive: If True, force the amplitude of sky components to be non-negative
    """

    if sky_dict is None:
        sky_file = os.path.join(user_params()['project_path'],f'sky_{instrument}/APERO_HE_Reference_A.fits')
        sp_sky = fits.getdata(sky_file,'SCI_SKY')
        wave_sky = fits.getdata(sky_file,'WAVE')
        regid_sky = fits.getdata(sky_file,'REG_ID')
        weights_sky = fits.getdata(sky_file,'WEIGHTS')
        sky_dict = dict(SCI_SKY=sp_sky, WAVE=wave_sky, REG_ID=regid_sky, WEIGHTS=weights_sky)
        return sky_dict

    spectrum_sky = wavecore.wave_to_wave(spectrum,wave,sky_dict['WAVE'])

    map_order = np.zeros_like(wave)
    for iord in range(spectrum.shape[0]):
        map_order[iord] = iord

    sky_recon= np.zeros_like(spectrum_sky)
    for regid in np.unique(sky_dict['REG_ID']):
        if regid ==0:
            continue

        mask = (sky_dict['REG_ID'] == regid)
        line_segment = spectrum_sky[mask]
        sp_sky_segment = sky_dict['SCI_SKY'][mask]

        g = (sky_dict['REG_ID'] == regid)
        weight = sky_dict['WEIGHTS'][g]
        order_line = map_order[g]
        glitch = (np.diff(order_line) !=0)
        if np.sum(glitch) !=0:
            glitch = np.concatenate(([False],glitch))
            line_segment[glitch] = np.nan
            sp_sky_segment[glitch] = np.nan

        def model_q(amp):
            return np.nansum( np.diff(line_segment -amp*sp_sky_segment*weight)**2)
        
        with warnings.catch_warnings(record=True) as _:
            guess = np.nanstd(np.diff(line_segment))
        x = minimize(model_q, guess)
        amp = x.x[0]    
        if force_positive and amp <0:
            amp = 0.0
        sky_recon[mask] = amp*sp_sky_segment*weight

    sky_recon = wavecore.wave_to_wave(sky_recon, sky_dict['WAVE'], wave)
    sky_recon[~np.isfinite(sky_recon)] = 0.0

    return sky_recon

def hotstar(hdr):
    hot_star_list = ['HD195094','HR1903','HR4023','HR3131','HR6743','HR7590',  'HR8709','HR9098','HR3117','HR3314','HR4467']
    
    drsobj = hdr['DRSOBJN'].strip()

    if drsobj in hot_star_list:
        hdr['HOTSTAR'] = True
    else:
        hdr['HOTSTAR'] = False
    return hdr


def load_calib(instrument: str):

    """Load instrument-dependent calibration and default TAPAS file."""
    if instrument == 'NIRPS':
        fwhm = fits.getdata(os.path.join(user_params()['project_path'],f'calib_{instrument}/C7A164F31A_pp_e2dsff_A_waveref_res_e2ds_A.fits'),'E2DS_FWHM')
        expo = fits.getdata(os.path.join(user_params()['project_path'],f'calib_{instrument}/C7A164F31A_pp_e2dsff_A_waveref_res_e2ds_A.fits'),'E2DS_EXPO')
        tfile = os.path.join(user_params()['project_path'],'LaSilla_tapas.fits')

    elif instrument == 'SPIROU':
        fwhm = fits.getdata(os.path.join(user_params['project_path'],f'calib_{instrument}/3444961B5Da_pp_e2dsff_AB_waveref_res_e2ds_AB.fits'),'E2DS_FWHM')
        expo = fits.getdata(os.path.join(user_params()['project_path'],f'calib_{instrument}/3444961B5Da_pp_e2dsff_AB_waveref_res_e2ds_AB.fits'),'E2DS_EXPO')
        tfile = os.path.join(user_params()['project_path'],'MaunaKea_tapas.fits')
    else:
        raise ValueError(f'Unknown instrument: {instrument}')
    return fwhm, expo, tfile

import warnings
# import SkyCoord
from astropy.coordinates import SkyCoord, EarthLocation


def optimize_exponents(wave, sp, airmass, fixed_exponents=None, guess=None, knee=user_params()['knee']):
    """
    Optimize telluric absorption exponents for H2O, CO2, CH4, O2.
    
    Parameters
    ----------
    wave : array (N, M)
        Wavelength grid
    sp : array (N, M)
        Spectrum to fit
    airmass : float
        Airmass of observation
    fixed_exponents : list of 4 or None
        List with None for free exponents, float for fixed ones
        Order: [H2O, CO2, CH4, O2]
    guess : list of 4 or None
        Initial guess for exponents
    knee : float
        Absorption threshold parameter
        
    Returns
    -------
    expo_optimal : list
        Optimized exponents [H2O, CO2, CH4, O2]
    """

    # Build initial exponents
    if fixed_exponents is None:
        expos_input = [airmass] * 4
    else:
        expos_input = [airmass if fe is None else fe for fe in fixed_exponents]

    # Precompute absorption reference
    all_abso = construct_abso(wave, expos=expos_input)
    abso0 = np.nanprod(all_abso, axis=0)
    trans_ref = construct_abso(wave, expos=expos_input, all_abso=all_abso)
    
    # Precompute masks and weights (constants for optimization)
    relevant = (trans_ref >= knee * 0.5) & (trans_ref <= 0.95)
    ww = weight_fall(trans_ref, knee=knee)
    wave_mask_inv = ~((wave >= wave_fit[0]) & (wave <= wave_fit[1]) & (abso0 > knee / 5.0))
    
    # Precompute pixel-to-pixel RMS (vectorized)
    pix2pixrms = np.nanmedian(np.abs(np.diff(sp, axis=1)), axis=1)
    pix2pixrms_30 = 30.0 * pix2pixrms[:, np.newaxis]  # shape (N, 1) for broadcasting
    
    # Build indices for variable exponents
    if fixed_exponents is None:
        var_indices = [0, 1, 2, 3]
    else:
        var_indices = [i for i in range(4) if fixed_exponents[i] is None]
    
    print('Starting exponent optimization...')
    
    def optimize_expo(variable_expos):
        # Reconstruct full exponents list
        if fixed_exponents is None:
            expos = list(variable_expos)
        else:
            expos = list(fixed_exponents)
            for j, i in enumerate(var_indices):
                expos[i] = variable_expos[j]
        
        # Compute transmission model
        trans2 = construct_abso(wave, expos=expos, all_abso=all_abso)
        trans2[trans2 < knee] = np.nan
        
        # Corrected spectrum
        corr = sp / trans2
        corr[wave_mask_inv] = np.nan
        
        # Gradient weighted by transmission
        grad = np.gradient(corr, axis=1) * weight_fall(trans2, knee=knee) * trans2
        
        # Objective: mean weighted gradient
        val_sum = np.nanstd(grad) #nanmean((grad * ww)[relevant]**2)
        
        # Progress output
        strout = ' '.join(f'{mol}: {exp:.4f}' for mol, exp in zip(molecules, expos))
        print(f'\r{strout} | Sum gradient^2: {val_sum:.8f}', end='', flush=True)
        
        return val_sum

    # Build initial guess and bounds
    if guess is None:
        x0_full = [1.0, airmass, airmass, airmass]
        bounds_full = [(0.001, 20), (airmass * 0.9, airmass * 1.1), 
                       (airmass * 0.9, airmass * 1.1), (airmass * 0.9, airmass * 1.1)]
    else:
        x0_full = []
        bounds_full = []
        for i in range(4):
            if guess[i] is None:
                x0_full.append(1.0 if i == 0 else airmass)
                bounds_full.append((0.1, 20) if i == 0 else (airmass * 0.9, airmass * 1.1))
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
        result = minimize(optimize_expo, x0=x0, bounds=bounds, 
                         method='Nelder-Mead', tol=5e-4)
    
    print()  # newline after progress
    
    # Reconstruct full exponents list
    if fixed_exponents is None:
        expo_optimal = list(result.x)
    else:
        expo_optimal = list(fixed_exponents)
        for j, i in enumerate(var_indices):
            expo_optimal[i] = result.x[j]

    return expo_optimal

"""
def optimize_exponents(wave, sp, airmass, fixed_exponents = None, guess = None, knee = user_params()['knee']):
    
    #if fixed_exponents is None, we fit all four exponents
    #otherwise we fit only the non-None exponents, keeping the others fixed
    #fixed_exponents: list of 4 elements, one per molecule, with None for free
    #                 exponents, and float for fixed exponents
    #0: H2O
    #1: CO2
    #2: CH4
    #3: O2

    if fixed_exponents is None:
        expos_input = [airmass]*4
    else:
        expos_input = []
        for i in range(4):
            if fixed_exponents[i] is None:
                expos_input.append(airmass)
            else:
                expos_input.append(fixed_exponents[i])

    all_abso = construct_abso(wave, expos=expos_input)
    abso0 = np.nanprod(all_abso, axis=0)

    trans_ref = construct_abso(wave, expos=expos_input, all_abso=all_abso)
    relevant = (trans_ref >= knee*0.5)*(trans_ref <= 0.95)
    ww = weight_fall(trans_ref,knee=knee)
    
    wave_mask = (wave >= wave_fit[0]) & (wave <= wave_fit[1]) & (abso0>knee/5.0)

    pix2pixrms = np.zeros(wave.shape[0])
    for iord in range(wave.shape[0]):
        pix2pixrms[iord] = np.nanmedian(np.abs(np.diff(sp[iord])))


    print('Starting exponent optimization...')
    def optimize_expo(variable_expos):
        expos = fixed_exponents
        # add the variable exponents in the right place
        if fixed_exponents is None:
            expos = variable_expos
        else:
            expos = []
            var_idx = 0
            for i in range(4):
                if fixed_exponents[i] is None:
                    expos.append(variable_expos[var_idx])
                    var_idx += 1
                else:
                    expos.append(fixed_exponents[i])

        trans2 = construct_abso(wave, expos=expos, all_abso=all_abso)
        trans2[trans2<0.03] = np.nan
        with warnings.catch_warnings(record=True) as _:
            corr = sp / trans2
        with warnings.catch_warnings(record=True) as _:
            corr[~wave_mask] = np.nan

        # do a filtering with savgol to remove low freq
        #corr = savgol_filter(corr, 9, 2, axis=1)

        #grad = np.diff(corr, axis=1)
        
        grad = np.gradient(corr, axis=1)*trans2

        with warnings.catch_warnings(record=True) as _:
            n1,p1= np.nanpercentile(grad,[16,84],axis=1)
        snr = (p1 - n1)/2.0
        snr[snr == 0] = np.nan
        snr_map = np.zeros_like(grad)

        for iord in range(grad.shape[0]):
            bad = np.abs(grad[iord])>(30*pix2pixrms[iord]/trans2[iord])
            #print(iord,np.sum(bad),pix2pixrms[iord])
            grad[iord][bad] = np.nan


        for iord in range(grad.shape[0]):
            snr_map[iord,:] = grad[iord]/snr[iord]

    
        # w_outlier = np.exp(-0.5*(snr_map/10)**2)

        grad[grad ==0]=np.nan
        
        grad_snr = np.abs(grad)
        val_sum = np.nanmean((grad_snr*ww)[relevant]) 

        # make a fancy print
        strout = ''
        for molecule, expo in zip(molecules, expos):
            strout += f'{molecule}: {expo:.4f} '
        print(f'\r{strout}| Sum gradient^2: {val_sum:.8f}',end="",flush = True)
        return val_sum

    if guess is None:
        x0_default = [1.0, airmass, airmass, airmass]
        bounds_default = [(0.001,20),(airmass*0.9, airmass*1.1),(airmass*0.9, airmass*1.1),(airmass*0.9, airmass*1.1)]
    else:
        x0_default = np.zeros_like(guess)
        for i in range(4):
            if guess[i] is None:
                if i ==0:
                    x0_default[i] = 1.0
                    bounds_default = [(0.1,20)]
                else:
                    x0_default[i] = airmass
                    bounds_default.append((airmass*0.9, airmass*1.1))
            else:
                x0_default[i] = guess[i]
                if i ==1:
                    bounds_default = [(guess[i]*0.9, guess[i]*1.1)]
                else:
                    bounds_default.append((guess[i]*0.9, guess[i]*1.1))
                

    if fixed_exponents is None:
        bounds = bounds_default
        x0 = x0_default
    else:
        bounds = []
        x0 = []
        for i in range(4):
            if fixed_exponents[i] is None:
                bounds.append(bounds_default[i])
                x0.append(x0_default[i])

    expo_optimal = minimize(optimize_expo, x0=x0, bounds=bounds, method='Nelder-Mead',tol=5e-4).x
    #expo_optimal = vals_x[-1]a

    #expo_optimal = minimize(optimize_expo, x0=x0, bounds=bounds, method='L-BFGS-B',    options={
    #    'ftol': 1e-9,      # tolérance sur la fonction (mettre très petit)
    #    'gtol': 1e-9,      # tolérance sur le gradient (mettre très petit)  
    #    'eps': 5e-4,       # pas pour l'approximation du gradient
    #    'maxiter': 100
    #}).x

    if fixed_exponents is not None:
        # reconstruct full expos list
        expos_full = []
        var_idx = 0
        for i in range(4):
            if fixed_exponents[i] is None:
                expos_full.append(expo_optimal[var_idx])
                var_idx += 1
            else:
                expos_full.append(fixed_exponents[i])
        expo_optimal = expos_full

    return expo_optimal
"""

def accurate_airmass(hdr):
    lat = hdr['BC_LAT']
    lon = hdr['BC_LONG']
    height = hdr['BC_ALT']  # in meters

    site = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=height*u.m)

    mjd_obs = hdr['MJDMID']
    time = Time(mjd_obs, format='mjd')

    altaz_frame = AltAz(obstime=time, location=site)
    # TODO -- use the true telescope pointing
    ra = hdr['PP_RA']
    dec = hdr['PP_DEC']

    obj_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    obj_altaz = obj_coord.transform_to(altaz_frame)

    # airmass accounting for atmosphere curvature
    z = 90.0 - obj_altaz.alt.degree
    R_earth = 6371.0  # km
    H_atmo = 8.43     # km
    z_rad = np.radians(z)
    sec_z = 1.0/np.cos(z_rad)
    am = np.sqrt((R_earth/(R_earth + H_atmo))**2 * sec_z**2 - (R_earth/(R_earth + H_atmo))**2 + 1.0)

    return am


def weight_fall(x, knee, w=None):
    """
    Smooth transition function: 0 -> 0.5 -> 1
    
    Parameters:
    x : array - input values
    x0 : float - midpoint (where function = 0.5)
    w : float - transition width (approximate distance from 0.1 to 0.9)
    """
    if w is None:
        w = knee/5.0

    return ne.evaluate('1 / (1 + exp(-4 * (x - knee) / w))')

def delay_since_sunset(mjd_obs, hdr=None):

    lat = hdr['BC_LAT']
    lon = hdr['BC_LONG']
    height = hdr['BC_ALT']  # in meters
    site = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=height*u.m)

    # Example date (from your MJDMID data)
    time = Time(mjd_obs, format='mjd')

    # Find sunset by checking when sun altitude crosses 0 degrees
    # Check times around the date
    delta_hours = np.linspace(0, 24, 1000)
    times = time + delta_hours * u.hour

    altaz_frame = AltAz(obstime=times, location=site)
    sun_altaz = get_sun(times).transform_to(altaz_frame)

    # Find when sun crosses horizon (altitude = 0)
    # Sunset is when altitude goes from positive to negative
    altitudes = sun_altaz.alt.degree
    sunset_idx = np.where((altitudes[:-1] > 0) & (altitudes[1:] < 0))[0]

    if len(sunset_idx) > 0:
        sunset_time = times[sunset_idx[0]]
        print(f'Sunset at observatory: {sunset_time.iso}')
        print(f'Sunset MJD: {sunset_time.mjd}')
    else:
        print('No sunset found in this 24h period')

    dt = mjd_obs - sunset_time.mjd
    dt_hours = dt * 24.0
    if dt_hours < 0:
        dt_hours += 24.0
    return dt_hours

def super_gauss_fast(xvector: np.ndarray, ew: np.ndarray, expo: np.ndarray):
    """
    Super gaussian using numexpr if possible

    ew = (fwhm/2)/(2*np.log(2))**(1/expo)
    supergauss = exp(-0.5*abs(xvector/ew_string)**expo)

    :param xvector: the xvector points to calculate the super gaussian from
    :param fwhm: np.ndarray, the full width half max value for the gaussian
                 at each pixel position
    :param expo: np.ndarray, the exponent of the super gaussian at each pixel
                 position

    :return: np.ndarray, the super gaussian profile for xvector, fwhm and expo
    """
    # if we have numexpr use it to do this fast
    # do not need to reference these using numexpr
    _ = xvector, expo
    # need to write as a string
    calc_string = f'exp(-0.5*abs(xvector/ew)**expo)'
    # evaluate string in numexpr
    return ne.evaluate(calc_string)

def variable_res_conv(wavemap: np.ndarray, spectrum: np.ndarray,
                      res_fwhm: np.ndarray, res_expo: np.ndarray,
                      ker_thres: float = 1e-4, ) -> np.ndarray:
    """
    Convolve with a variable kernel in resolution space

    :param wavemap: np.ndarray, wavelength grid
    :param spectrum: np.ndarray, spectrum to be convolved
    :param res_fwhm: np.ndarray, fwhm at each pixel, same shape as wave and
                     spectrum
    :param res_expo: np.ndarray, expoenent parameter for the PSF, expo=2
                     would be gaussian
    :param ker_thres: float, optional, amplitude of kernel at which we stop
                      convolution

    :return: np.ndarray, the convolved spectrum
    """
    # get shape of spectrum (2D or 1D)
    shape0 = spectrum.shape
    # -------------------------------------------------------------------------
    # if we have an e2ds ravel
    if len(shape0) == 2:
        res_fwhm = res_fwhm.ravel()
        res_expo = res_expo.ravel()
        wavemap = wavemap.ravel()
        spectrum = spectrum.ravel()
    # -------------------------------------------------------------------------
    # convolved outputs
    sumker = np.zeros_like(spectrum)
    spectrum2 = np.zeros_like(spectrum)
    # -------------------------------------------------------------------------
    # get the width of the scanning of the kernel. Default is 3 FWHM
    scale1 = np.max(res_fwhm)
    scale2 = np.median(np.gradient(wavemap) / wavemap) * speed_of_light
    range_scan = 20 * (scale1 / scale2)
    # round scan range to pixel level
    range_scan = int(np.ceil(range_scan))
    # mask nan pixels
    valid_pix = np.isfinite(spectrum)
    # set to zero the pixels that are NaNs
    spectrum[~valid_pix] = 0.0
    # convert non valid pixels to floats
    valid_pix = valid_pix.astype(float)
    # sorting by distance to center of kernel
    range2 = np.arange(-range_scan, range_scan)
    range2 = range2[np.argsort(abs(range2))]
    # calculate the super gaussian width
    ew = (res_fwhm / 2) / (2 * np.log(2)) ** (1 / res_expo)
    # -------------------------------------------------------------------------
    # loop around each offset scanning the sum and constructing local kernels
    for offset in range2:
        # get the dv offset
        dv = speed_of_light * (wavemap / np.roll(wavemap, offset) - 1)
        # calculate the kernel at this offset
        ker = super_gauss_fast(dv, ew, res_expo)
        # stop convolving when threshold reached
        if np.max(ker) < ker_thres:
            break
        # no weight if the pixel was a NaN value
        ker = ker * valid_pix
        # add this kernel to the convolved spectrum
        spectrum2 = spectrum2 + np.roll(spectrum, offset) * ker
        # save the kernel
        sumker = sumker + ker
    # -------------------------------------------------------------------------
    # normalize convovled spectrum to kernel sum
    with warnings.catch_warnings(record=True) as _:
        spectrum2 = spectrum2 / sumker
    # reshape if necessary
    if len(shape0) == 2:
        spectrum2 = spectrum2.reshape(shape0)
    # return convolved spectrum
    return spectrum2


def mask_o2(wave):
    transm_file = '../LaSilla_NIRPS_tapas.fits'
    transm_table = Table.read(transm_file)

    trans_o2 = np.array(transm_table['O2']  )

    lines = np.where((np.gradient(np.gradient(trans_o2))>0) & (trans_o2<0.8) & (trans_o2<np.roll(trans_o2,1)) & (trans_o2<np.roll(trans_o2,-1)))

    mask = np.zeros_like(transm_table['wavelength'], dtype=bool)
    velostep = (np.nanmedian(np.gradient(transm_table['wavelength'])/transm_table['wavelength']*3e5)*1000)
    window_size = 8000 # in m/s
    for dv in range(-int(window_size/velostep), int(window_size/velostep)+1):
        mask[lines[0]+dv] = True

    spl_o2 = ius(transm_table['wavelength'], mask.astype(float), ext=0,k=1)

    return spl_o2(wave) >0.5



def construct_abso(wave, expos, all_abso = None):
    molecules = ['H2O','CO2', 'CH4',  'O2']
    transm_file = '../LaSilla_NIRPS_tapas.fits'

    if all_abso is None:
        transm_table = Table.read(transm_file)
        # exponents are for 'H2O', 'CH4', 'CO2', 'O2'
        # molecules_to_merge = [ 'O2', 'O3','NO2','N2O']

        tbl = Table()
        tbl['wavelength'] = transm_table['wavelength']
        tbl['H2O'] = transm_table['H2O']
        tbl['CH4'] = transm_table['CH4']
        tbl['CO2'] = transm_table['CO2']
        tbl['O2'] = transm_table['O2'] # already merged
        transm_table = tbl
        keep_wave = (transm_table['wavelength'] >= np.min(wave)) & (transm_table['wavelength'] <= np.max(wave))
        transm_table = transm_table[keep_wave]

        transm_wave = np.array(transm_table['wavelength'])
        molecules =  np.array(transm_table.keys())[1:]

        all_abso = np.zeros((len(molecules), wave.shape[0], wave.shape[1]))
        for i, molecule in enumerate(molecules):
            all_abso[i] = ius(transm_wave, transm_table[molecule], ext=0,k=1)(wave)
            all_abso[i][all_abso[i] < 0] = 0.0

        return all_abso


    absos = np.ones_like(wave)

    # Build the expression string dynamically
    expr_parts = [f'a{i}**e{i}' for i in range(len(molecules))]
    expr = 'absos * ' + ' * '.join(expr_parts)

    # Build the local dictionary
    local_dict = {'absos': absos}
    for i in range(len(molecules)):
        local_dict[f'a{i}'] = all_abso[i]
        local_dict[f'e{i}'] = expos[i]

    # Single numexpr evaluation
    absos = ne.evaluate(expr, local_dict=local_dict)

    #for i, molecule in enumerate(molecules):
    #    absos *= all_abso[i] ** expos[i]

    trans2 = variable_res_conv(wave, absos, E2DS_FWHM, E2DS_EXPO)
    trans2[trans2<user_params()['knee']/5.0] = np.nan
   
    return trans2

def fetch_template(hdr, wavemin=None, wavemax=None):
    """
    Fetch and interpolate a stellar template spectrum based on header parameters.
    
    This function creates a stellar spectrum by interpolating between
    two temperature models based on the stellar effective temperature. The template
    is corrected for systemic velocity and wavelength-calibrated for the specific
    instrument.
    
    Args:
        hdr: FITS header containing stellar and instrumental parameters
        
    Returns:
        spline: InterpolatedUnivariateSpline object for the stellar template
    """
    # TODO --> we should include vsini in the interpolation as well at some point

    # Extract effective temperature from pipeline processing
    teff = hdr['PP_TEFF']

    # Fallback to ESO header if pipeline RV is zero
    # This handles cases where the pipeline didn't compute RV
    if hdr['PP_RV'] ==0 and hdr['HIERARCH ESO TEL TARG RADVEL'] !=0:
        print("Using HIERARCH ESO TEL TARG RADVEL for systemic velocity")
        hdr['PP_RV'] = hdr['HIERARCH ESO TEL TARG RADVEL']
    
    # Set wavelength range based on instrument
    # Different instruments have different wavelength coverage
    if  hdr['INSTRUME'].upper() == 'NIRPS':
        if wavemin is None:
            wave1 = 950   # Minimum wavelength in nm
        else:
            wave1 = wavemin
        if wavemax is None:
            wave2 = 2000  # Maximum wavelength in nm
        else:
            wave2 = wavemax
    elif hdr['INSTRUME'].upper() == 'SPIROU':
        if wavemin is None:
            wave1 = 950   # Minimum wavelength in nm
        else:
            wave1 = wavemin
        if wavemax is None:
            wave2 = 2550  # Maximum wavelength in nm
        else:
            wave2 = wavemax
    else:
        # Raise error for unsupported instruments
        raise ValueError(f"Unknown instrument {hdr['INSTRUME']}")


    # Round effective temperature to nearest 500K bracket
    # Templates are available in 500K steps from 3000K to 6000K
    t_up = int(teff/500 + 1) * 500    # Upper temperature bracket
    t_low = t_up - 500                 # Lower temperature bracket

    # Enforce temperature limits of the model grid
    if t_low < 3000:
        t_low = 3000  # Minimum available model temperature
    if t_up > 6000:
        t_up = 6000   # Maximum available model temperature

    # Construct filenames for the two bracketing temperature models
    filename_low = os.path.join(user_params()['project_path'], f'models/temperature_gradient_{t_low}.fits')
    filename_up = os.path.join(user_params()['project_path'], f'models/temperature_gradient_{t_up}.fits')

    # Calculate linear interpolation weights
    # weight_up + weight_low = 1.0
    weight_up = (teff - t_low) / (t_up - t_low)
    weight_low = 1.0 - weight_up

    # Read the two temperature model files
    tbl_low = Table.read(filename_low)
    tbl_up = Table.read(filename_up)

    # Extract wavelength array and apply systemic velocity correction
    # This shifts the template to the target's rest frame
    wave = np.array(tbl_low['wavelength'])

    # Extract flux arrays from both models
    log_flux_low = np.array(tbl_low['flux'])
    log_flux_up = np.array(tbl_up['flux'])

    # Linearly interpolate flux between the two temperature models
    log_flux = log_flux_low * weight_low + log_flux_up * weight_up

    dv = np.gradient(log_flux)/np.gradient(wave)*speed_of_light

    # Filter wavelength range to instrument coverage and remove non-finite values
    keep = (wave >= wave1) & (wave <= wave2) & np.isfinite(log_flux) & np.isfinite(dv)


    # Apply the filter mask
    wave = wave[keep]
    flux = np.exp(log_flux[keep])
    dv = dv[keep]
    # Create a cubic spline interpolator for resampling the template
    # k=3 means cubic interpolation, ext=1 means extrapolation returns zero
    spline = ius(wave, flux, k=1, ext=3)
    spline_dv = ius(wave, dv, k=1, ext=3)   

    

    return spline, spline_dv


# Make calib globals available to helper functions
E2DS_FWHM, E2DS_EXPO, transm_file = load_calib(instrument)
#global E2DS_FWHM, E2DS_EXPO, transm_file
