"""
Slinky wavelength solution refinement tools.

Ported from the slinky repository and adapted for the telluric_fit pipeline.
Uses telluric_config.yaml + batch_config.yaml for configuration.
Scientific data (FP/HC line lists, wave solutions) are discovered from
calib_INSTRUMENT/ directories that must be synced beforehand (see sync_NIRPS,
sync_SPIROU).

Author: Etienne Artigau (original slinky code)
Adaptation: 2026-04
"""

import glob
import os
from shutil import copyfile
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.constants import c
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from tqdm import tqdm
import yaml

from tellu_tools_config import tprint, get_project_path


# ============================================================================
# YAML helpers
# ============================================================================

def load_yaml(yaml_file):
    with open(yaml_file, 'r') as fh:
        return yaml.safe_load(fh)


# ============================================================================
# Instrument-specific defaults
# ============================================================================

_INSTRUMENT_DEFAULTS = {
    'NIRPS': {
        'fiber': 'A',
        'inst_wavestart': 965,
        'inst_waveend': 1950,
        'WAVEFILE_KEY': 'WAVEFILE',
    },
    'SPIROU': {
        'fiber': 'AB',
        'inst_wavestart': 965,
        'inst_waveend': 2500,
        'WAVEFILE_KEY': 'WAVEFILE',
    },
}


def _get_slinky_params(config):
    """
    Build the full slinky parameter dict from the telluric_config.yaml content
    and machine-detected paths.

    Parameters
    ----------
    config : dict
        Content of telluric_config.yaml

    Returns
    -------
    params : dict
    """
    instrument = config.get('instrument', 'NIRPS').upper()
    if instrument not in _INSTRUMENT_DEFAULTS:
        raise ValueError(f"Instrument {instrument} not supported. "
                         f"Choose from {list(_INSTRUMENT_DEFAULTS)}")

    defaults = _INSTRUMENT_DEFAULTS[instrument]
    project_path = get_project_path()

    slinky_cfg = config.get('slinky', {})

    params = {
        'instrument': instrument,
        'fiber': defaults['fiber'],
        'inst_wavestart': defaults['inst_wavestart'],
        'inst_waveend': defaults['inst_waveend'],
        'WAVEFILE_KEY': defaults['WAVEFILE_KEY'],
        'doplot': slinky_cfg.get('doplot', False),
        'calib_dir': os.path.join(project_path, f'calib_{instrument}'),
        'patched_wavesol': os.path.join(project_path, f'calib_{instrument}_patched'),
        'output_slinky': os.path.join(project_path, f'slinky_{instrument}_output'),
        'plot_folder': os.path.join(project_path, f'slinky_{instrument}_plots'),
        'hot_stars': config.get('hot_stars', []),
        'science_targets': config.get('science_targets', []),
        'wave_leverage': slinky_cfg.get('wave_leverage', 1600),
        'wslinky': slinky_cfg.get('wslinky', 1e-1),
    }

    # Create directories as needed
    for key in ('patched_wavesol', 'output_slinky', 'plot_folder'):
        os.makedirs(params[key], exist_ok=True)

    return params


# ============================================================================
# Utility functions (ported from slinky.py / etienne_tools.py)
# ============================================================================

def val_cheby(coeffs, xvector, domain):
    """Evaluate a Chebyshev polynomial on *xvector* mapped to *domain*."""
    domain_cheby = 2 * (xvector - domain[0]) / (domain[1] - domain[0]) - 1
    return np.polynomial.chebyshev.chebval(domain_cheby, coeffs)


def gp_project(x, y, yerr, wslinky=1e-1, xmin=980, xmax=1850, npts=100000):
    """Project scattered data onto a regular grid with a Gaussian kernel."""
    tprint(f'    GP projection: {len(x)} points sur {npts} pixels (wslinky={wslinky} nm)...', color='green')
    xv = np.linspace(xmin, xmax, npts)
    weights = np.full(npts, 1e-12)
    yv = np.zeros(npts)

    xvbis = xv / wslinky
    xbis = x / wslinky
    for i in tqdm(range(len(x)), leave=False, desc='GP kernel'):
        dd = xvbis - xbis[i]
        g = np.abs(dd) < 10
        dd2 = dd[g]
        w2 = np.exp(-0.5 * dd2 ** 2) / yerr[i] ** 2
        weights[g] += w2
        yv[g] += w2 * y[i]
    yv /= weights
    return xv, yv


def odd_ratio_mean(value, err, odd_ratio=1e-4, nmax=10):
    """Iterative weighted mean with outlier rejection (odd-ratio method)."""
    keep = np.isfinite(value) * np.isfinite(err)
    if np.sum(keep) == 0:
        return np.nan, np.nan
    value = value[keep]
    err = err[keep]
    guess = np.nanmedian(value)
    for _ in range(nmax):
        nsig = (value - guess) / err
        gg = np.exp(-0.5 * nsig ** 2)
        odd_bad = odd_ratio / (gg + odd_ratio)
        odd_good = 1 - odd_bad
        w = odd_good / err ** 2
        guess = np.nansum(value * w) / np.nansum(w)
    bulk_error = np.sqrt(1 / np.nansum(odd_good / err ** 2))
    return guess, bulk_error


def odd_ratio_linfit(x, y, yerr):
    """Iterative weighted linear fit with outlier rejection."""
    g = np.isfinite(y + yerr + x)
    x, y, yerr = x[g], y[g], yerr[g]
    w = np.ones(len(x))
    wsum = 1.0
    wsum0 = 0.0
    while np.abs(wsum0 - wsum) > 1e-6:
        wsum0 = np.sum(w)
        fit, sig = np.polyfit(x, y, 1, w=w / yerr, cov=True)
        errfit = np.sqrt(np.diag(sig))
        res = (y - np.polyval(fit, x)) / yerr
        p1 = np.exp(-0.5 * res ** 2)
        w = p1 / (p1 + 1e-6)
        wsum = np.sum(w)
    return fit, errfit


def sigma(v):
    """Robust standard deviation (half-width of 68 % CI)."""
    n1, p1 = np.nanpercentile(v, [16, 84])
    return 0.5 * (p1 - n1)


def search_fits_with_mjd(search_string, mjdkey='MJDMID'):
    """Glob *search_string*, return files sorted by MJD."""
    files = glob.glob(search_string)
    if not files:
        return np.array([]), np.array([])
    tprint(f'    Lecture des entêtes MJD : {len(files)} fichiers ({os.path.basename(search_string)})...', color='green')
    mjds = np.full(len(files), np.nan)
    for i, f in enumerate(tqdm(files, leave=False, desc='MJD headers')):
        try:
            mjds[i] = fits.getheader(f)[mjdkey]
        except (OSError, KeyError):
            tprint(f'    Avertissement : fichier ignoré (corrompu ou MJD absent) : {f}', color='yellow')
    valid = np.isfinite(mjds)
    files = np.array(files)[valid]
    mjds = mjds[valid]
    order = np.argsort(mjds)
    return files[order], mjds[order]


def mjd_to_matplotlib_date(mjd):
    """Convert MJD to matplotlib date (days since 1858-11-17 shifted)."""
    return mjd - 40587.50


# ============================================================================
# Core slinky: refine_wavesol
# ============================================================================

def refine_wavesol(params):
    """
    Refine wavelength solutions using the slinky (cavity) method.

    Reads FP and HC line-list FITS files from *params['calib_dir']*.
    Produces patched wavelength solutions in *params['patched_wavesol']*.

    Parameters
    ----------
    params : dict
        As returned by :func:`_get_slinky_params`.
    """
    wave_leverage = params['wave_leverage']
    wslinky = params['wslinky']
    doplot = params['doplot']
    calib_dir = params['calib_dir']
    patched_dir = params['patched_wavesol']
    plot_folder = params['plot_folder']
    fiber = params['fiber']
    inst_wavestart = params['inst_wavestart']
    inst_waveend = params['inst_waveend']
    instrument = params['instrument']

    # ------------------------------------------------------------------
    # Locate reference files
    # ------------------------------------------------------------------
    ref_files_hc = glob.glob(f'{calib_dir}/*waveref_hclines*{fiber}.fits')
    if not ref_files_hc:
        raise FileNotFoundError(
            f"No HC reference file (*waveref_hclines*{fiber}.fits) in {calib_dir}. "
            "Run sync_NIRPS / sync_SPIROU first."
        )
    ref_file_hc = ref_files_hc[0]
    tbl_hc_ref = Table.read(ref_file_hc)

    wavesol_files = glob.glob(f'{calib_dir}/*_wavesol_ref_{fiber}.fits')
    if not wavesol_files:
        raise FileNotFoundError(
            f"No wavesol ref file (*_wavesol_ref_{fiber}.fits) in {calib_dir}. "
            "Run sync_NIRPS / sync_SPIROU first."
        )
    ref_wave_sol = fits.getdata(wavesol_files[0])

    # ------------------------------------------------------------------
    # Find all FP / HC files, match by MJD
    # ------------------------------------------------------------------
    files_fp, mjds_fp = search_fits_with_mjd(f'{calib_dir}/*wave_fplines_{fiber}.fits')
    files_hc, mjds_hc = search_fits_with_mjd(f'{calib_dir}/*wave_hclines_{fiber}.fits')

    if len(files_fp) == 0 or len(files_hc) == 0:
        raise FileNotFoundError(
            f"No FP ({len(files_fp)}) or HC ({len(files_hc)}) files found in {calib_dir}. "
            "Run sync_NIRPS / sync_SPIROU first."
        )

    tprint(f'[SLINKY] {len(files_fp)} FP and {len(files_hc)} HC files found', color='green')

    # Keep only FP files that have a matching HC within 0.1 day
    dt = np.array([np.min(np.abs(m - mjds_hc)) for m in mjds_fp])
    g = dt < 0.1
    files_fp = files_fp[g]
    mjds_fp = mjds_fp[g]

    tprint(f'[SLINKY] {len(files_fp)} FP files matched to HC', color='green')

    # ------------------------------------------------------------------
    # Step 1 – Compute cavity for each HC epoch
    # ------------------------------------------------------------------
    tprint(f'[SLINKY] Étape 1/4 : calcul de la cavité pour {len(files_hc)} époques HC...', color='cyan')
    for i_hc in range(len(files_hc)):
        file_hc_updated = files_hc[i_hc].replace('.fits', '_slinky.fits')
        if os.path.isfile(file_hc_updated):
            tprint(f'  HC {i_hc+1}/{len(files_hc)} already processed, skipping', color='yellow')
            continue

        tprint(f'  Processing HC {i_hc+1}/{len(files_hc)}', color='green')
        tbl_hc = Table.read(files_hc[i_hc], 'WAVE_HCLIST')

        tbl_hc_tmp = Table(tbl_hc_ref)
        ii = np.zeros(len(tbl_hc_tmp), dtype=int)
        for iline in tqdm(range(len(tbl_hc_tmp)), leave=False):
            g = ((tbl_hc_tmp['ORDER'][iline] == tbl_hc['ORDER']) *
                 (tbl_hc_tmp['WAVE_REF'][iline] == tbl_hc['WAVE_REF']))
            if np.sum(g) == 0:
                continue
            ii[iline] = np.where(g)[0][0]

        tbl_hc = Table(tbl_hc[ii])

        if 'CAVITY' in tbl_hc.colnames:
            tprint(f'  Cavity already in table, skipping', color='yellow')
            continue

        # Find matching FP
        i_fp = np.argmin(np.abs(mjds_fp - mjds_hc[i_hc]))
        if np.abs(mjds_fp[i_fp] - mjds_hc[i_hc]) > 0.5:
            tprint(f'  No matching FP for HC {i_hc+1}, skipping', color='red')
            continue

        tbl_fp = Table.read(files_fp[i_fp], 'WAVE_FPLIST')

        tbl_hc['CAVITY'] = np.nan
        mask = tbl_hc['PIXEL_MEAS'].mask
        all_frac_peak = np.zeros_like(tbl_hc['PIXEL_MEAS'])
        all_cavity = np.zeros_like(tbl_hc['PIXEL_MEAS'])
        wave_ref = np.array(tbl_hc['WAVE_REF'])
        pixel_meas = np.array(tbl_hc['PIXEL_MEAS'])
        current_order = -1

        for i in tqdm(range(len(tbl_hc)), leave=False):
            if mask[i]:
                continue
            order = tbl_hc['ORDER'][i]
            if order != current_order:
                tbl_fp_order = tbl_fp[tbl_fp['ORDER'] == order]
                tbl_fp_order = tbl_fp_order[~tbl_fp_order['PIXEL_MEAS'].mask]
                current_order = order
                spl = ius(tbl_fp_order['PIXEL_MEAS'], tbl_fp_order['PEAK_NUMBER'], k=1, ext=1)
            all_frac_peak[i] = spl(pixel_meas[i])
            all_cavity[i] = wave_ref[i] * all_frac_peak[i]

        bad = all_frac_peak == 0
        all_frac_peak[bad] = np.nan
        all_cavity[bad] = np.nan

        tbl_hc['PEAK_NUMBER'] = all_frac_peak
        tbl_hc['CAVITY'] = all_cavity

        for col in tbl_hc.colnames:
            try:
                tbl_hc[col][tbl_hc[col].mask] = np.nan
            except Exception:
                pass

        copyfile(files_hc[i_hc], file_hc_updated)
        with fits.open(file_hc_updated) as hdul:
            hdul[1].data = tbl_hc.as_array()
            hdul.writeto(file_hc_updated, overwrite=True)

    # ------------------------------------------------------------------
    # Step 2 – Build cavity statistics across epochs
    # ------------------------------------------------------------------
    tprint(f'[SLINKY] Étape 2/4 : statistiques de cavité sur {len(files_hc)} époques...', color='cyan')
    tbl_hc_ref = Table.read(ref_file_hc)
    href = fits.getheader(ref_file_hc)
    cavity_polynomial = np.array([href[key] for key in href['WCAV0*'].keys()])
    WCAV_PED = href['WCAV_PED']

    order = np.argsort(mjds_hc)
    mjds_hc = mjds_hc[order]
    files_hc = files_hc[order]

    all_cavity = np.zeros([len(files_hc), len(tbl_hc_ref)], dtype=float)
    for ifile, file in tqdm(enumerate(files_hc), leave=False):
        file_hc_updated = file.replace('.fits', '_slinky.fits')
        all_cavity[ifile] = Table.read(file_hc_updated, 'WAVE_HCLIST')['CAVITY'].data.data

    all_cavity[all_cavity == 0] = np.nan
    med_per_line = np.nanmedian(all_cavity, axis=0)

    meds = np.zeros(len(files_hc))
    for iepoch in range(len(files_hc)):
        meds[iepoch] = np.nanmedian(all_cavity[iepoch] - med_per_line)
        all_cavity[iepoch] -= meds[iepoch]

    plt.plot(mjds_hc, meds, '.')
    plt.savefig(f'{plot_folder}/cavity_median_{instrument}.pdf')
    plt.close()

    med_per_line = np.nanmedian(all_cavity, axis=0)

    domain = [inst_wavestart, inst_waveend]
    cavity_ref = val_cheby(cavity_polynomial, tbl_hc_ref['WAVE_REF'], domain=domain) + WCAV_PED
    dv_ref = c * (med_per_line / cavity_ref - 1)
    bad = np.abs(dv_ref) > 1000
    dv_ref[bad] = np.nan
    tbl_hc_ref['WAVE_REF'] = tbl_hc_ref['WAVE_REF'] * (1 - dv_ref / c)
    med_per_line[bad] = np.nan

    n1, p1 = np.nanpercentile(all_cavity, [16, 84], axis=0)
    sig_per_line = (p1 - n1) / 2
    bad = np.sum(np.isfinite(all_cavity), axis=0) < all_cavity.shape[0] // 5
    sig_per_line[sig_per_line == 0] = np.nan
    sig_per_line[bad] = np.nan
    sig_per_line_ms = c * (sig_per_line / med_per_line)

    # ------------------------------------------------------------------
    # Step 3 – Measure zero-point & slope per HC epoch
    # ------------------------------------------------------------------
    tprint(f'[SLINKY] Étape 3/4 : mesure du zéro-point et pente pour {len(files_hc)} époques HC...', color='cyan')
    all_slopes = np.zeros(len(files_hc), dtype=float)
    all_errslopes = np.zeros_like(all_slopes)
    all_pedestals = np.zeros_like(all_slopes)
    all_errpedestals = np.zeros_like(all_slopes)

    fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
    ax[0].plot_date([np.nan], [np.nan])
    ax[1].plot_date([np.nan], [np.nan])

    for ifile, file in enumerate(files_hc):
        file_hc_updated = file.replace('.fits', '_slinky.fits')
        tbl2 = Table.read(file_hc_updated, 'WAVE_HCLIST')
        h = fits.getheader(file_hc_updated)

        dcavity = c * (tbl2['CAVITY'].data.data / med_per_line - 1)
        sdcavity = c * sig_per_line / med_per_line
        wave2 = tbl2['WAVE_REF'] / 1e3 - wave_leverage / 1e3

        moy, err = odd_ratio_mean(dcavity, sdcavity)
        fit, sig_fit = odd_ratio_linfit(wave2, dcavity - moy, sdcavity)
        fit[1] += moy

        tprint(f'  HC {ifile+1}/{len(files_hc)} zp {fit[1]:5.2f}+-{sig_fit[1]:5.2f} m/s, '
               f'slope {fit[0]:5.2f}+-{sig_fit[0]:5.2f} m/s/um', color='green')

        if doplot:
            ax[0].errorbar(mjd_to_matplotlib_date(h['MJDMID']), fit[1], yerr=sig_fit[1], fmt='.g')
            ax[1].errorbar(mjd_to_matplotlib_date(h['MJDMID']), fit[0], yerr=sig_fit[0], fmt='.g')

        all_slopes[ifile] = fit[0]
        all_errslopes[ifile] = sig_fit[0]
        all_pedestals[ifile] = fit[1]
        all_errpedestals[ifile] = sig_fit[1]

    ax[0].grid(color='grey', linestyle='--', linewidth=0.5)
    ax[1].grid(color='grey', linestyle='--', linewidth=0.5)
    ax[0].set(ylabel='Zero-point [m/s]')
    ax[1].set(ylabel='Slope [m/s/µm]', xlabel='Date')
    if doplot:
        plt.show()
    else:
        plt.savefig(f'{plot_folder}/wavesol_{instrument}.png')
        plt.close()

    tprint(f'  Consecutive pedestal sigma: '
           f'{sigma(all_pedestals - np.roll(all_pedestals, 1)) / np.sqrt(2):.3f} m/s',
           color='green')

    # ------------------------------------------------------------------
    # Step 4 – Patch each FP wavelength solution
    # ------------------------------------------------------------------
    tprint(f'[SLINKY] Étape 4/4 : correction de {len(files_fp)} solutions FP...', color='cyan')
    recovered_pedestal = np.zeros(len(files_fp), dtype=float)
    recovered_slope = np.zeros_like(recovered_pedestal)
    recovered_errslope = np.zeros_like(recovered_pedestal)
    recovered_errpedestal = np.zeros_like(recovered_pedestal)

    for i_fp in range(len(files_fp)):
        tprint(f'  Patching FP wavesol {i_fp+1}/{len(files_fp)}', color='green')
        file_fp = files_fp[i_fp]
        hdr = fits.getheader(file_fp)
        wavefile = os.path.join(calib_dir, hdr['WAVEFILE'])
        patched_wavefile = os.path.join(patched_dir, hdr['WAVEFILE'])

        hdr_fp = fits.getheader(file_fp)
        tbl_fp = Table.read(file_fp, 'WAVE_FPLIST')

        i_hc = np.argmin(np.abs(mjds_hc - hdr_fp['MJDMID']))
        slope_hc = all_slopes[i_hc]
        err_slope_hc = all_errslopes[i_hc]
        pedestal_hc = all_pedestals[i_hc]
        err_pedestal_hc = all_errpedestals[i_hc]

        if os.path.isfile(patched_wavefile):
            tprint(f'    Already patched, skipping', color='yellow')
            continue

        # Apply cavity correction
        wavelength_model = np.array(tbl_fp['WAVE_REF'].data)
        doppler_shift = (wavelength_model / 1e3 - wave_leverage / 1e3) * slope_hc + pedestal_hc
        cavity = val_cheby(cavity_polynomial, wavelength_model,
                           domain=[inst_wavestart, inst_waveend]) + WCAV_PED
        peak_number = tbl_fp['PEAK_NUMBER'].data
        wavelength_model = cavity / peak_number * (1 + doppler_shift / c)
        tbl_fp['WAVE_REF'] = wavelength_model

        wavemap = np.array(ref_wave_sol)
        dv_residuals = []
        dv_residuals_err = []
        wave_ref_residuals = []

        for order in np.unique(tbl_fp['ORDER']):
            g = tbl_fp['ORDER'] == order
            tbl_order = tbl_fp[g]
            pixel_meas = np.array(tbl_order['PIXEL_MEAS'])
            wave_ref = np.array(tbl_order['WAVE_REF'])
            valid = np.isfinite(pixel_meas + wave_ref)
            pixel_meas, wave_ref = pixel_meas[valid], wave_ref[valid]

            fit = np.polyfit(pixel_meas, wave_ref, 5)
            residual = (wave_ref / np.polyval(fit, pixel_meas) - 1) * c
            wave_order = np.polyval(fit, np.arange(4088))

            if np.all(np.isfinite(wave_order)):
                wavemap[order] = wave_order

            err = sigma(residual - np.roll(residual, 1)) / np.sqrt(2)
            dv_residuals.append(residual)
            dv_residuals_err.append(np.ones_like(residual) * err)
            wave_ref_residuals.append(wave_ref)

        dv_residuals = np.concatenate(dv_residuals)
        dv_residuals_err = np.concatenate(dv_residuals_err)
        wave_ref_residuals = np.concatenate(wave_ref_residuals)

        npts = int((np.nanmax(wavemap) - np.nanmin(wavemap)) / wslinky * 5)
        xmin, xmax = np.nanmin(wavemap), np.nanmax(wavemap)

        xv, yv = gp_project(wave_ref_residuals, dv_residuals, dv_residuals_err,
                             wslinky=wslinky, xmin=xmin, xmax=xmax, npts=npts)
        spl = ius(xv, yv, k=2)
        wavemap = wavemap * (1 + spl(wavemap) / c)

        # Sanity check: back-project HC lines
        i_hc_check = np.argmin(np.abs(hdr_fp['MJDMID'] - mjds_hc))
        file_hc_updated = files_hc[i_hc_check].replace('.fits', '_slinky.fits')
        tbl_hc = Table.read(file_hc_updated)
        tbl_hc['WAVE_REF'] = tbl_hc_ref['WAVE_REF']

        dv_fp = []
        dv_hc = []
        wave_hc_list = []
        hc_meas = np.array(tbl_hc['PIXEL_MEAS'])

        fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=True)

        for iord in np.unique(tbl_hc['ORDER']):
            g = tbl_hc['ORDER'] == iord
            spl_ord = ius(np.arange(4088), wavemap[iord], k=2)
            hc_meas[g] = spl_ord(tbl_hc['PIXEL_MEAS'][g])
            v = np.array(tbl_hc['WAVE_REF'][g] / spl_ord(tbl_hc['PIXEL_MEAS'][g]) - 1) * c
            dv_hc.append(v)
            wave_hc_list.append(tbl_hc['WAVE_REF'][g])
            if doplot:
                ax2[0].plot(tbl_hc['WAVE_REF'][g], v, 'o', alpha=0.1)

            g2 = tbl_fp['ORDER'] == iord
            v2 = np.array(tbl_fp['WAVE_REF'][g2] / spl_ord(tbl_fp['PIXEL_MEAS'][g2]) - 1) * c
            if doplot:
                ax2[1].plot(tbl_fp['WAVE_REF'][g2], v2, 'o', alpha=0.1)
            dv_fp.append(v2)

        dv_hc_plot = np.concatenate(dv_hc)
        dv_fp_plot = np.concatenate(dv_fp)
        wave_hc_plot = np.concatenate(wave_hc_list)

        tprint(f'    RMS FP: {sigma(dv_fp_plot):.2f} m/s', color='green')

        sig_per_line_ms_plot = np.array(sig_per_line_ms)
        ord_hc = np.argsort(wave_hc_plot)
        dv_hc_plot = dv_hc_plot[ord_hc]
        wave_hc_plot = wave_hc_plot[ord_hc]
        sig_per_line_ms_plot = sig_per_line_ms_plot[ord_hc]

        mean_hcs, err_hcs, mean_waves = [], [], []
        for i in range(len(dv_hc_plot) // 100):
            mh, sh = odd_ratio_mean(dv_hc_plot[i*100:(i+1)*100],
                                    sig_per_line_ms_plot[i*100:(i+1)*100])
            mw = np.mean(wave_hc_plot[i*100:(i+1)*100])
            ax2[0].errorbar(mw, mh, yerr=sh, fmt='k.')
            mean_hcs.append(mh)
            err_hcs.append(sh)
            mean_waves.append(mw)

        mean_hcs = np.array(mean_hcs)
        err_hcs = np.array(err_hcs)
        mean_waves = np.array(mean_waves)
        valid = np.isfinite(mean_hcs) * np.isfinite(err_hcs) * np.isfinite(mean_waves)
        mean_hcs, err_hcs, mean_waves = mean_hcs[valid], err_hcs[valid], mean_waves[valid]

        fit2, cov2 = np.polyfit((mean_waves - wave_leverage) / 1000, mean_hcs, 1,
                                w=1 / err_hcs, cov=True)
        sig2 = np.sqrt(np.diag(cov2))
        tprint(f'    Residual slope: {fit2[0]:5.2f}+-{sig2[0]:5.2f} m/s/um, '
               f'pedestal: {fit2[1]:5.2f}+-{sig2[1]:5.2f} m/s', color='green')

        ax2[0].set(ylabel='RV [m/s]', ylim=[-30, 30])
        ax2[1].set(xlabel='Wavelength [nm]', ylabel='RV [m/s]', ylim=[-30, 30])
        ax2[0].grid(color='grey', linestyle='--', linewidth=0.5)
        ax2[1].grid(color='grey', linestyle='--', linewidth=0.5)
        if doplot:
            plt.show()
        else:
            plt.savefig(f'{plot_folder}/fp_hc_{instrument}_{i_fp}.png')
            plt.close()

        dv_hc_final = np.array((hc_meas / tbl_hc['WAVE_REF'] - 1) * c)
        valid = np.isfinite(dv_hc_final)
        fit_r, sig_r = odd_ratio_linfit(
            (tbl_hc['WAVE_REF'] - wave_leverage)[valid] / 1e3,
            dv_hc_final[valid], sig_per_line_ms[valid])

        recovered_pedestal[i_fp] = fit_r[1]
        recovered_slope[i_fp] = fit_r[0]
        recovered_errslope[i_fp] = sig_r[0]
        recovered_errpedestal[i_fp] = sig_r[1]

        tprint(f'    RMS HC: {sigma(dv_hc_final):.2f} m/s  RMS FP: {sigma(dv_fp_plot):.2f} m/s',
               color='green')

        # Write patched wavesol
        copyfile(wavefile, patched_wavefile)
        with fits.open(patched_wavefile) as hdul:
            hdul[1].data = wavemap
            hdul[0].header['SLINKY'] = (True, 'Wavelength solution corrected for cavity effect')
            hdul[0].header['ZPCAV'] = (pedestal_hc, 'Zero-point [m/s]')
            hdul[0].header['ZPCAVER'] = (err_pedestal_hc, 'Error on zero-point [m/s]')
            hdul[0].header['SLPCAV'] = (slope_hc, 'Slope [m/s/um]')
            hdul[0].header['SLPCAVER'] = (err_slope_hc, 'Error on slope [m/s/um]')
            hdul.writeto(patched_wavefile, overwrite=True)
        tprint(f'    Wrote {patched_wavefile}', color='green')

    # Summary plot
    tprint(f'  Overall RMS:  pedestal {np.std(recovered_pedestal):.3f} m/s, '
           f'slope {np.std(recovered_slope):.3f} m/s/um', color='green')

    fig3, ax3 = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax3[0].errorbar(mjds_fp, recovered_pedestal, yerr=recovered_errpedestal, fmt='g.')
    ax3[1].errorbar(mjds_fp, recovered_slope, yerr=recovered_errslope, fmt='g.')
    ax3[0].set(ylabel='Zero-point [m/s]')
    ax3[1].set(ylabel='Slope [m/s/µm]', xlabel='MJD')
    if doplot:
        plt.show()
    else:
        plt.savefig(f'{plot_folder}/wavesol_{instrument}_patched.png')
        plt.close()


# ============================================================================
# Core slinky: padding_wavesol
# ============================================================================

def padding_wavesol(params, science_files=None):
    """
    Replace the WaveA/WaveAB extension in science FITS files with the
    patched (slinky-corrected) wavelength solution.

    Parameters
    ----------
    params : dict
        As returned by :func:`_get_slinky_params`.
    science_files : list of str, optional
        Explicit list of science FITS files to patch. If *None*, uses
        all ``*_e2dsff_{fiber}.fits`` in ``scidata_INSTRUMENT/`` and
        ``hotstars_INSTRUMENT/``.
    """
    instrument = params['instrument']
    fiber = params['fiber']
    patched_dir = params['patched_wavesol']
    output_dir = params['output_slinky']
    WAVEFILE_KEY = params['WAVEFILE_KEY']
    hot_stars = params.get('hot_stars', [])
    science_targets = params.get('science_targets', [])

    project_path = get_project_path()

    if science_files is None:
        # Hot stars are always included
        patterns = [
            os.path.join(project_path, f'hotstars_{instrument}', f'*_e2dsff_{fiber}.fits'),
        ]
        # If science_targets is set, only include those specific targets
        if science_targets:
            for target in science_targets:
                patterns.append(
                    os.path.join(project_path, f'scidata_{instrument}', target,
                                 f'*_e2dsff_{fiber}.fits'))
        else:
            # No target list: include all science data
            patterns.append(
                os.path.join(project_path, f'scidata_{instrument}', '**',
                             f'*_e2dsff_{fiber}.fits'))
        science_files = []
        for pat in patterns:
            science_files.extend(glob.glob(pat, recursive=True))

    if not science_files:
        tprint(f'[SLINKY] No science files found to patch for {instrument}', color='red')
        return

    all_wave_sol_files = np.array(glob.glob(os.path.join(patched_dir, '*.fits')))
    if len(all_wave_sol_files) == 0:
        tprint(f'[SLINKY] No patched wavesol files in {patched_dir}. Run refine_wavesol first.',
               color='red')
        return

    tprint(f'[SLINKY] Padding {len(science_files)} science files with slinky wavesol', color='green')

    for ifile, file in enumerate(science_files):
        try:
            tprint(f'  Padding {ifile+1}/{len(science_files)}: {os.path.basename(file)}', color='green')
            hdr0 = fits.getheader(file, ext=0)
            hdr1 = fits.getheader(file, ext=1)
            hdr = {**dict(hdr1), **dict(hdr0)}  # ext=0 prend priorité
            if WAVEFILE_KEY not in hdr:
                tprint(f'    Clé "{WAVEFILE_KEY}" absente de l\'entête, fichier ignoré', color='yellow')
                continue
            wavefile = hdr[WAVEFILE_KEY]

            keep = np.array([wavefile in w for w in all_wave_sol_files])
            if True not in keep:
                tprint(f'    No patched wavesol for {wavefile}, skipping', color='red')
                continue

            # Determine output path
            basename = os.path.basename(file)
            outname_slinky = os.path.join(output_dir,
                                          basename.replace('.fits', '_slinky.fits'))

            if os.path.exists(outname_slinky):
                tprint(f'    Already exists, skipping', color='yellow')
                continue

            wave_sol_file = all_wave_sol_files[keep][0]
            wavesol = fits.getdata(wave_sol_file)

            copyfile(file, outname_slinky)
            hdu = fits.open(outname_slinky, mode='update')
            hdu[f'Wave{fiber}'].data = wavesol
            hdu.close()
            tprint(f'    Wrote {outname_slinky}', color='green')

        except Exception as e:
            tprint(f'    Error: {e}', color='red')
            continue


# ============================================================================
# Main entry points
# ============================================================================

def run_slinky(science_files=None):
    """
    Full slinky pipeline: refine wavelength solutions then patch science files.

    Reads configuration from telluric_config.yaml (resolved relative to this
    module's location, same as the rest of the pipeline).

    Parameters
    ----------
    science_files : list of str, optional
        Explicit list of science files to patch after refinement.
        If None, auto-discovered from scidata/hotstars directories.
    """
    tprint('[SLINKY] Chargement de tellu_tools...', color='green')
    import tellu_tools as tt
    config = tt.load_telluric_config()
    params = _get_slinky_params(config)
    tprint(f'[SLINKY] Instrument : {params["instrument"]}', color='green')
    tprint(f'[SLINKY] Calib dir  : {params["calib_dir"]}', color='green')
    tprint(f'[SLINKY] Patched dir: {params["patched_wavesol"]}', color='green')
    tprint(f'[SLINKY] --- Raffinement des solutions de longueur d\'onde ---', color='cyan')
    refine_wavesol(params)
    tprint(f'[SLINKY] --- Padding des fichiers science ---', color='cyan')
    padding_wavesol(params, science_files=science_files)
    tprint('[SLINKY] Terminé.', color='green')


if __name__ == '__main__':
    run_slinky()
