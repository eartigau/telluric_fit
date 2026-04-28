#!/usr/bin/env python
"""
Spectrum Inspection Tool for NIRPS and SPIROU Data

Creates PDF plots comparing telluric-corrected spectra to templates,
showing TAPAS atmospheric transmission and OH emission lines.

Adapted from https://github.com/eartigau/spectrum_inspector
Extended to support tellupatched_t.fits files from the telluric correction pipeline.

Usage:
    python spectrum_inspector.py spectrum.fits                    # All orders PDF
    python spectrum_inspector.py spectrum.fits template.fits      # With explicit template
    python spectrum_inspector.py spectrum.fits --order 50         # Single order
    python spectrum_inspector.py spectrum.fits --order 40 60      # Orders 40-60
    python spectrum_inspector.py --batch tellupatched_NIRPS/TOI4552_v1/  # Batch mode

Supported file types:
    - APERO t.fits files
    - ESO r.*.fits files  
    - tellupatched_t.fits files (from this pipeline)

Author: Adapted from spectrum_inspector by Étienne Artigau
Date: 2026-02-05
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from datetime import datetime
from typing import Optional, Dict, Tuple, List
import argparse
import glob


# ============================================================================
# Utility Functions
# ============================================================================

def doppler(wave: np.ndarray, velocity: float) -> np.ndarray:
    """Apply Doppler shift to wavelength array."""
    c = 299792.458  # speed of light in km/s
    return wave * np.sqrt((1 - velocity / c) / (1 + velocity / c))


def robust_polyfit(x: np.ndarray, y: np.ndarray, degree: int = 4,
                   sigma_clip: float = 3.0, max_iter: int = 10) -> np.ndarray:
    """
    Perform robust polynomial fit with iterative sigma clipping.

    Parameters
    ----------
    x : array
        Independent variable
    y : array
        Dependent variable
    degree : int
        Polynomial degree
    sigma_clip : float
        Number of sigma for outlier rejection
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    fit_values : array
        Polynomial evaluated at x positions
    """
    mask = np.isfinite(x) & np.isfinite(y)
    fit = np.ones_like(y) * np.nan

    for _ in range(max_iter):
        if np.sum(mask) < degree + 1:
            return np.ones_like(y) * np.nan

        coeffs = np.polyfit(x[mask], y[mask], degree)
        fit = np.polyval(coeffs, x)

        residuals = y - fit
        sigma = np.nanstd(residuals[mask])

        if sigma == 0:
            break

        new_mask = np.isfinite(x) & np.isfinite(y) & (np.abs(residuals) < sigma_clip * sigma)

        if np.array_equal(mask, new_mask):
            break

        mask = new_mask

    return fit


def get_file_type(fits_file: str) -> Optional[str]:
    """
    Determine file type from filename.

    Returns
    -------
    str or None
        'APERO' for t.fits files
        'ESO' for r.*.fits files
        'TELLUPATCHED' for tellupatched_t.fits files
        None if unrecognized
    """
    basename = os.path.basename(fits_file)

    if 'tellupatched_t.fits' in basename:
        return 'TELLUPATCHED'
    elif fits_file.endswith('t.fits'):
        return 'APERO'
    elif basename.startswith('r.'):
        return 'ESO'
    else:
        return None


# ============================================================================
# Data Loading Functions
# ============================================================================

def get_oh_lines(wave_min: float, wave_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """Download and return OH emission lines in the specified range."""
    os.makedirs('reference_data', exist_ok=True)
    oh_file = 'reference_data/tablea1.dat'

    url = 'https://cdsarc.cds.unistra.fr/ftp/J/A+A/581/A47/table1.dat'
    if not os.path.exists(oh_file):
        import urllib.request
        print(f"Downloading OH line list from {url}...")
        urllib.request.urlretrieve(url, oh_file)
        print("  Done.")

    oh_lines = Table.read(oh_file, format='ascii')
    wave_oh = np.concatenate([
        np.array(oh_lines['col1'].data),
        np.array(oh_lines['col3'].data)
    ]) / 10.0  # convert Angstrom to nm
    label_oh = np.concatenate([oh_lines['col2'].data, oh_lines['col4'].data])

    keep = (wave_oh >= wave_min) & (wave_oh <= wave_max)
    return wave_oh[keep], label_oh[keep]


def load_tapas(tapas_file: str = 'reference_data/tapas_lbl.fits') -> Table:
    """Load TAPAS atmospheric transmission data."""
    if not os.path.exists(tapas_file):
        os.makedirs('reference_data', exist_ok=True)

        url = 'http://206.12.93.77/ari/data/lbl/tapas/tapas_lbl.fits'
        print(f"TAPAS file '{tapas_file}' not found locally.")
        print(f"Attempting to download from: {url}")

        try:
            import urllib.request
            urllib.request.urlretrieve(url, tapas_file)

            if os.path.exists(tapas_file):
                file_size = os.path.getsize(tapas_file)
                if file_size < 1000:
                    os.remove(tapas_file)
                    raise ValueError(f"Downloaded file is too small ({file_size} bytes).")
                print(f"✓ Successfully downloaded tapas_lbl.fits ({file_size / (1024*1024):.1f} MB)")
            else:
                raise FileNotFoundError("Download completed but file not found.")

        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download TAPAS file from {url}\n"
                f"Error: {e}\n\n"
                f"Please copy 'tapas_lbl.fits' from your LBL 'models/' folder."
            )

    return Table.read(tapas_file)


def load_template(template_file: str) -> Tuple:
    """Load template spectrum and return interpolator."""
    tbl_template = Table(fits.getdata(template_file))
    wave_template = tbl_template['wavelength'].data
    flux_template = tbl_template['flux'].data

    keep = np.isfinite(wave_template) & np.isfinite(flux_template)

    return (IUS(wave_template[keep], flux_template[keep], k=1),
            IUS(wave_template, np.array(keep, dtype=float), k=1))


def load_spectrum(fits_file: str, instrument: str = 'NIRPS') -> Dict:
    """Load spectrum data from FITS file."""
    if instrument == 'NIRPS':
        fiber_setup = 'A'
    elif instrument == 'SPIROU':
        fiber_setup = 'AB'
    else:
        raise ValueError("Instrument must be either 'NIRPS' or 'SPIROU'.")

    file_type = get_file_type(fits_file)

    # ESO files have different header structure
    if file_type == 'ESO':
        hdr_primary = fits.getheader(fits_file, ext=0)
        hdr = fits.getheader(fits_file, ext=1)
        berv = hdr_primary['ESO QC BERV']
        try:
            syst_vel = hdr_primary['ESO TEL TARG RADVEL']
        except KeyError:
            syst_vel = np.nan
    else:
        hdr = fits.getheader(fits_file, ext=1)
        berv = hdr.get('BERV', 0.0)
        try:
            syst_vel = hdr['ESO TEL TARG RADVEL']
        except KeyError:
            syst_vel = np.nan

    # Load data based on file type
    if file_type == 'ESO':
        flux = fits.getdata(fits_file, 'SCIDATA')
        wave = fits.getdata(fits_file, 'WAVEDATA_VAC_BARY') / 10.0  # Angstrom to nm
        blaze = np.ones_like(flux)
        wave_in_rest_frame = True
    else:
        # APERO and TELLUPATCHED files
        flux = fits.getdata(fits_file, f'Flux{fiber_setup}')
        try:
            blaze = fits.getdata(fits_file, f'Blaze{fiber_setup}')
        except KeyError:
            blaze = np.ones_like(flux)
        wave = fits.getdata(fits_file, f'Wave{fiber_setup}')
        wave_in_rest_frame = False

    # Convert zeros to NaN
    flux[flux == 0] = np.nan

    # Normalize blaze
    for iord in range(flux.shape[0]):
        blaze[iord] /= np.nanpercentile(blaze[iord], 95)

    # Extract metadata
    if file_type == 'ESO':
        target = hdr_primary.get('OBJECT', 'Unknown')
        date_obs = hdr_primary.get('DATE-OBS', datetime.now().strftime('%Y-%m-%d'))
    else:
        target = hdr.get('OBJECT', 'Unknown')
        date_obs = hdr.get('DATE-OBS', datetime.now().strftime('%Y-%m-%d'))

    return {
        'flux': flux,
        'blaze': blaze,
        'wave': wave,
        'berv': berv,
        'syst_vel': syst_vel,
        'v_tot': berv - syst_vel if np.isfinite(syst_vel) else np.nan,
        'target': target,
        'date_obs': date_obs,
        'instrument': instrument,
        'n_orders': flux.shape[0],
        'fits_file': os.path.basename(fits_file),
        'file_type': file_type,
        'wave_in_rest_frame': wave_in_rest_frame,
        'needs_blaze_correction': (file_type == 'ESO' and '_BLAZE_' in os.path.basename(fits_file)),
    }


def find_template(fits_file: str, instrument: str = 'NIRPS') -> str:
    """
    Auto-detect template file based on object name from FITS header.
    """
    hdr = fits.getheader(fits_file, ext=1)
    obj_name = hdr.get('DRSOBJN', hdr.get('OBJECT', 'Unknown')).replace(' ', '_')

    if instrument == 'NIRPS':
        fiber = 'A'
    else:
        fiber = 'AB'

    template_name = f'Template_s1dv_{obj_name}_sc1d_v_file_{fiber}.fits'

    if os.path.exists(template_name):
        print(f"Using auto-detected template: {template_name}")
        return template_name
    else:
        raise FileNotFoundError(
            f"Template file not found. Looked for: {template_name}\n"
            f"Please provide the template file as the second argument."
        )


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_order(spectrum_data: Dict, template_interp: Tuple, order: int,
               tapas: Optional[Table] = None,
               wave_oh: Optional[np.ndarray] = None,
               label_oh: Optional[np.ndarray] = None,
               show_oh: bool = True, show_tapas: bool = True) -> Tuple:
    """
    Plot a single spectral order.

    Parameters
    ----------
    spectrum_data : dict
        Dictionary with flux, blaze, wave, berv keys
    template_interp : tuple
        Interpolator for template spectrum
    order : int
        Order number to plot
    tapas : Table, optional
        TAPAS atmospheric data
    wave_oh, label_oh : array, optional
        OH line positions and labels
    show_oh : bool
        Whether to show OH lines
    show_tapas : bool
        Whether to show TAPAS transmission

    Returns
    -------
    fig, ax : matplotlib figure and axes, or (None, None) if order is invalid
    """
    wave = spectrum_data['wave']
    flux = spectrum_data['flux']
    blaze = spectrum_data['blaze']
    berv = spectrum_data['berv']
    wave_in_rest_frame = spectrum_data.get('wave_in_rest_frame', False)
    needs_blaze_correction = spectrum_data.get('needs_blaze_correction', False)

    wave_ord = wave[order]

    # Check if flux is all NaN
    if np.all(~np.isfinite(flux[order])):
        return None, None

    # Blaze correction
    flux_blaze_corrected = flux[order] / blaze[order]

    flux_median = np.nanmedian(flux_blaze_corrected)
    if not np.isfinite(flux_median) or flux_median == 0:
        return None, None

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8), nrows=2, sharex=True)

    # Normalize flux
    plot_tmp = flux_blaze_corrected / flux_median

    # Get template at correct velocity
    if wave_in_rest_frame:
        template_tmp = template_interp[0](wave_ord)
        template_mask = template_interp[1](wave_ord) > 0.5
    else:
        template_tmp = template_interp[0](doppler(wave_ord, -berv))
        template_mask = template_interp[1](doppler(wave_ord, -berv)) > 0.5
    template_tmp[~template_mask] = np.nan
    template_tmp /= np.nanmedian(template_tmp)

    # Store original for background plot
    plot_tmp_original = None
    flux_label = 'Normalized Flux'

    # Apply additional blaze correction for ESO files
    if needs_blaze_correction:
        ratio = plot_tmp / template_tmp
        blaze_fit = robust_polyfit(wave_ord, ratio, degree=4, sigma_clip=3.0, max_iter=10)

        if np.any(np.isfinite(blaze_fit)):
            plot_tmp_original = plot_tmp.copy()
            plot_tmp = plot_tmp / blaze_fit
            plot_tmp /= np.nanmedian(plot_tmp[np.isfinite(plot_tmp)])
            flux_label = 'Normalized Flux (blaze-corrected)'

    # Find valid wavelength range
    flux_valid = np.isfinite(plot_tmp)
    if not np.any(flux_valid):
        plt.close(fig)
        return None, None

    valid_indices = np.where(flux_valid)[0]
    idx_min, idx_max = valid_indices[0], valid_indices[-1]
    wave_min, wave_max = wave_ord[idx_min], wave_ord[idx_max]

    # Plot spectra
    if plot_tmp_original is not None:
        ax[0].plot(wave_ord, plot_tmp_original, alpha=0.3, color='grey',
                   label='Original (uncorrected)', zorder=1, rasterized=True)

    ax[0].plot(wave_ord, plot_tmp, alpha=0.7, color='black', label=flux_label,
               zorder=2, rasterized=True)
    ax[0].plot(wave_ord, template_tmp, alpha=0.7, color='red', label='Template',
               zorder=3, rasterized=True)

    # Residuals
    residuals = plot_tmp - template_tmp
    ax[1].plot(wave_ord, residuals, color='orange', label='Residuals', rasterized=True)

    # Add OH lines
    if show_oh and wave_oh is not None:
        keep = (wave_oh >= wave_min) & (wave_oh <= wave_max)
        for w, lab in zip(wave_oh[keep], label_oh[keep]):
            ax[0].axvline(w, color='cyan', alpha=1.0, linewidth=1.0)
            ax[1].axvline(w, color='cyan', alpha=1.0, linewidth=1.0)
            ax[0].text(w, 0.05, lab, rotation=90, verticalalignment='bottom',
                       horizontalalignment='right', fontsize=5, alpha=0.5)

    # Add TAPAS
    if show_tapas and tapas is not None:
        keep = (tapas['WAVELENGTH'] >= wave_min) & (tapas['WAVELENGTH'] <= wave_max)
        tapas_trim = tapas[keep]
        if len(tapas_trim) > 0:
            ax[0].plot(tapas_trim['WAVELENGTH'], tapas_trim['ABSO_OTHERS'],
                       color='green', alpha=0.9, label='TAPAS Other Gases',
                       linewidth=1.0, rasterized=True)
            ax[0].plot(tapas_trim['WAVELENGTH'], tapas_trim['ABSO_WATER'],
                       color='blue', alpha=0.9, label='TAPAS Water',
                       linewidth=1.0, rasterized=True)

    ax[0].legend(loc='upper right', fontsize=8)
    ax[0].set_xlim(wave_min, wave_max)

    # Y-axis limits
    if plot_tmp_original is not None:
        ymax_flux = 1.2 * np.nanpercentile(plot_tmp_original, 90)
        ax[0].set_ylim(0, ymax_flux)
    else:
        ax[0].set_ylim(bottom=0)

    ax[1].set_xlabel('Wavelength (nm)')
    ax[0].set_ylabel('Normalized Flux')
    ax[1].set_ylabel('Residuals')

    # Residual y-limits
    res_valid = residuals[np.isfinite(residuals)]
    if len(res_valid) > 0:
        ymin, ymax = np.min(res_valid), np.max(res_valid)
        ymin = max(ymin, -0.5)
        ymax = min(ymax, 0.5)
        ylim = (np.abs(ymin) + np.abs(ymax)) / 2
        ax[1].set_ylim(-ylim, ylim)
    else:
        ax[1].set_ylim(-0.5, 0.5)

    # Title
    title = (f"{spectrum_data['instrument']} | {spectrum_data['target']} | "
             f"{spectrum_data['date_obs']} | Order {order} | {wave_min:.1f}-{wave_max:.1f} nm")
    ax[0].set_title(title)

    ax[0].grid(alpha=0.3)
    ax[1].grid(alpha=0.3)

    return fig, ax


def generate_pdf_name(spectrum_data: Dict, order_min: Optional[int] = None,
                      order_max: Optional[int] = None,
                      output_dir: Optional[str] = None) -> str:
    """Generate smart PDF filename."""
    timestamp = spectrum_data['date_obs'].replace(':', '-')

    parts = [
        spectrum_data['instrument'],
        spectrum_data['file_type'],
        spectrum_data['target'].replace(' ', '_'),
        timestamp,
    ]

    if order_min is not None and order_max is not None:
        if order_min == order_max:
            parts.append(f'order{order_min}')
        else:
            parts.append(f'orders{order_min}-{order_max}')
    else:
        parts.append('all_orders')

    filename = '_'.join(parts) + '.pdf'

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    return filename


def create_summary_page(spectrum_data: Dict, template_file: str, fits_file: str,
                        order_min: int, order_max: int):
    """Create a summary page with observation information."""
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, "Spectrum Inspection Report",
            fontsize=24, fontweight='bold', ha='center', va='top',
            transform=ax.transAxes)
    ax.text(0.5, 0.88, f"{spectrum_data['instrument']} - {spectrum_data['target']}",
            fontsize=18, ha='center', va='top', transform=ax.transAxes)

    # Left column
    syst_vel_str = f"{spectrum_data['syst_vel']:.3f} km/s" if np.isfinite(spectrum_data['syst_vel']) else "N/A"
    v_tot_str = f"{spectrum_data['v_tot']:.3f} km/s" if np.isfinite(spectrum_data['v_tot']) else "N/A"

    left_lines = [
        "OBSERVATION DETAILS",
        "=" * 35,
        "",
        f"Target:        {spectrum_data['target']}",
        f"Instrument:    {spectrum_data['instrument']}",
        f"Date:          {spectrum_data['date_obs']}",
        f"File type:     {spectrum_data['file_type']}",
        "",
        "",
        "FILES",
        "=" * 35,
        "",
        f"Spectrum:",
        f"  {os.path.basename(fits_file)}",
        "",
        f"Template:",
        f"  {os.path.basename(template_file)}",
    ]

    right_lines = [
        "VELOCITIES",
        "=" * 35,
        "",
        f"BERV:          {spectrum_data['berv']:.3f} km/s",
        f"Systemic (Vsys): {syst_vel_str}",
        f"V_tot (BERV-Vsys): {v_tot_str}",
        "",
        "",
        "SPECTRAL ORDERS",
        "=" * 35,
        "",
        f"Total orders:  {spectrum_data['n_orders']}",
        f"In this PDF:   {order_min} to {order_max}",
        f"               ({order_max - order_min + 1} orders)",
        "",
        f"Wavelength:    {np.nanmin(spectrum_data['wave']):.1f} - {np.nanmax(spectrum_data['wave']):.1f} nm",
    ]

    left_text = '\n'.join(left_lines)
    right_text = '\n'.join(right_lines)

    ax.text(0.25, 0.78, left_text, fontsize=12, fontfamily='monospace',
            ha='center', va='top', transform=ax.transAxes)
    ax.text(0.75, 0.78, right_text, fontsize=12, fontfamily='monospace',
            ha='center', va='top', transform=ax.transAxes)

    # Footer
    ax.text(0.5, 0.05, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=10, ha='center', va='bottom', transform=ax.transAxes, alpha=0.5)

    return fig


# ============================================================================
# Main Inspection Functions
# ============================================================================

def inspect_all_orders(fits_file: str, template_file: str,
                       tapas_file: str = 'reference_data/tapas_lbl.fits',
                       instrument: str = 'NIRPS',
                       order_min: Optional[int] = None,
                       order_max: Optional[int] = None,
                       show_oh: bool = True, show_tapas: bool = True,
                       output_dir: Optional[str] = None) -> str:
    """
    Create multi-page PDF with one page per spectral order.

    Parameters
    ----------
    fits_file : str
        Path to NIRPS/SPIROU FITS file
    template_file : str
        Path to template spectrum
    tapas_file : str
        Path to TAPAS file
    instrument : str
        'NIRPS' or 'SPIROU'
    order_min, order_max : int, optional
        Range of orders to plot
    show_oh : bool
        Show OH emission lines
    show_tapas : bool
        Show TAPAS transmission
    output_dir : str, optional
        Output directory for PDF. If None, saves in current directory.

    Returns
    -------
    pdf_path : str
        Path to saved PDF
    """
    # Load spectrum data
    spectrum_data = load_spectrum(fits_file, instrument)
    template_interp = load_template(template_file)
    n_orders = spectrum_data['n_orders']

    # Determine order range
    if order_min is None:
        order_min = 0
    if order_max is None:
        order_max = n_orders - 1
    order_min = max(0, order_min)
    order_max = min(n_orders - 1, order_max)

    # Load OH lines and TAPAS
    wave_oh, label_oh = None, None
    tapas = None

    if show_oh:
        full_wave_min = np.nanmin(spectrum_data['wave'])
        full_wave_max = np.nanmax(spectrum_data['wave'])
        wave_oh, label_oh = get_oh_lines(full_wave_min, full_wave_max)

    if show_tapas:
        tapas = load_tapas(tapas_file)

    # Generate PDF name
    pdf_path = generate_pdf_name(spectrum_data, order_min, order_max, output_dir)

    print(f"Creating PDF with orders {order_min} to {order_max} ({order_max - order_min + 1} orders + summary page)...")

    with PdfPages(pdf_path) as pdf:
        # First page: Summary
        fig_summary = create_summary_page(spectrum_data, template_file, fits_file,
                                          order_min, order_max)
        pdf.savefig(fig_summary, dpi=150)
        plt.close(fig_summary)

        # Order pages
        for order in range(order_min, order_max + 1):
            print(f"  Processing order {order}/{order_max}...", end='\r')

            fig, ax = plot_order(
                spectrum_data, template_interp, order,
                tapas=tapas, wave_oh=wave_oh, label_oh=label_oh,
                show_oh=show_oh, show_tapas=show_tapas
            )

            if fig is None:
                continue

            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"\nSaved: {pdf_path}")
    return pdf_path


def inspect_single_order(fits_file: str, template_file: str, order: int,
                         tapas_file: str = 'reference_data/tapas_lbl.fits',
                         instrument: str = 'NIRPS',
                         show_oh: bool = True, show_tapas: bool = True,
                         save_pdf: bool = True, show: bool = True,
                         output_dir: Optional[str] = None) -> Optional[str]:
    """
    Plot and optionally save a single spectral order.
    """
    spectrum_data = load_spectrum(fits_file, instrument)
    template_interp = load_template(template_file)

    wave_ord = spectrum_data['wave'][order]
    wave_min, wave_max = np.nanmin(wave_ord), np.nanmax(wave_ord)

    wave_oh, label_oh = get_oh_lines(wave_min, wave_max) if show_oh else (None, None)
    tapas = load_tapas(tapas_file) if show_tapas else None

    fig, ax = plot_order(
        spectrum_data, template_interp, order,
        tapas=tapas, wave_oh=wave_oh, label_oh=label_oh,
        show_oh=show_oh, show_tapas=show_tapas
    )

    if fig is None:
        print(f"Order {order} has no valid data, skipping.")
        return None

    plt.tight_layout()

    pdf_path = None
    if save_pdf:
        pdf_path = generate_pdf_name(spectrum_data, order, order, output_dir)
        fig.savefig(pdf_path, dpi=150)
        print(f"Saved: {pdf_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return pdf_path


def inspect_batch(batch_dir: str, instrument: str = 'NIRPS',
                  order_min: Optional[int] = None,
                  order_max: Optional[int] = None,
                  max_files: Optional[int] = None) -> List[str]:
    """
    Create inspection PDFs for all spectra in a batch folder.

    PDFs are saved to a parallel _plot folder:
        tellupatched_NIRPS/object_batch/ -> tellupatched_NIRPS/object_batch_plot/

    Parameters
    ----------
    batch_dir : str
        Path to batch folder containing tellupatched_t.fits files
    instrument : str
        'NIRPS' or 'SPIROU'
    order_min, order_max : int, optional
        Range of orders to include
    max_files : int, optional
        Maximum number of files to process

    Returns
    -------
    pdf_paths : list
        List of generated PDF paths
    """
    # Find all tellupatched files
    patterns = [
        os.path.join(batch_dir, '*tellupatched_t.fits'),
        os.path.join(batch_dir, '*t.fits'),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))

    if max_files is not None:
        files = files[:max_files]

    print(f"Found {len(files)} files in {batch_dir}")

    if not files:
        print("No files found.")
        return []

    # Create output directory (parallel _plot folder)
    base_dir = os.path.dirname(batch_dir.rstrip('/'))
    folder_name = os.path.basename(batch_dir.rstrip('/'))
    output_dir = os.path.join(base_dir, folder_name + '_plot')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Try to find template
    template_file = None
    if files:
        try:
            template_file = find_template(files[0], instrument)
        except FileNotFoundError:
            print("Warning: Could not auto-detect template.")
            print("Please provide template file as second argument.")
            return []

    pdf_paths = []
    for i, f in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {os.path.basename(f)}")
        try:
            pdf_path = inspect_all_orders(
                f, template_file,
                instrument=instrument,
                order_min=order_min, order_max=order_max,
                output_dir=output_dir
            )
            pdf_paths.append(pdf_path)
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'='*60}")
    print(f"Generated {len(pdf_paths)} PDFs in {output_dir}")
    print(f"{'='*60}")

    return pdf_paths


# ============================================================================
# Command-line Interface
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inspect NIRPS/SPIROU spectra by order',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python spectrum_inspector.py spectrum.fits template.fits
  python spectrum_inspector.py spectrum.fits                   # Auto-detect template
  python spectrum_inspector.py spectrum.fits template.fits --order 50
  python spectrum_inspector.py spectrum.fits template.fits --order 40 60
  python spectrum_inspector.py --batch tellupatched_NIRPS/TOI4552_v1/
"""
    )

    parser.add_argument('file', nargs='?', help='Input FITS file')
    parser.add_argument('template', nargs='?', default=None,
                        help='Template spectrum file (optional)')
    parser.add_argument('--batch', help='Batch directory to process')
    parser.add_argument('--tapas', default='reference_data/tapas_lbl.fits',
                        help='TAPAS transmission file')
    parser.add_argument('--instrument', default='NIRPS', choices=['NIRPS', 'SPIROU'],
                        help='Instrument name')
    parser.add_argument('--order', type=int, nargs='+',
                        help='Order(s) to plot. Single value or range.')
    parser.add_argument('--max-files', type=int, help='Max files for batch mode')
    parser.add_argument('--output-dir', help='Output directory for PDFs')
    parser.add_argument('--no-oh', action='store_true', help='Hide OH lines')
    parser.add_argument('--no-tapas', action='store_true', help='Hide TAPAS')
    parser.add_argument('--no-save', action='store_true', help='Do not save PDF')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot')

    args = parser.parse_args()

    # Batch mode
    if args.batch:
        inspect_batch(
            args.batch, instrument=args.instrument,
            order_min=args.order[0] if args.order and len(args.order) >= 1 else None,
            order_max=args.order[1] if args.order and len(args.order) >= 2 else (args.order[0] if args.order else None),
            max_files=args.max_files
        )
    elif args.file:
        # Check input file exists
        if not os.path.exists(args.file):
            print(f"Error: Input file '{args.file}' not found.")
            exit(1)

        # Check file type
        file_type = get_file_type(args.file)
        if file_type is None:
            print(f"\n{'='*70}")
            print("ERROR: Unrecognized file type!")
            print(f"{'='*70}")
            print(f"\nFile: {args.file}")
            print("\nThis script works with:")
            print("  1. APERO t.fits files (ends with 't.fits')")
            print("  2. ESO r.*.fits files (starts with 'r.')")
            print("  3. tellupatched_t.fits files")
            print(f"\n{'='*70}")
            exit(1)

        print(f"Detected file type: {file_type}")

        # Auto-detect template if not provided
        if args.template is None:
            if file_type == 'ESO':
                print("Error: Template file required for ESO files.")
                exit(1)
            try:
                args.template = find_template(args.file, args.instrument)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                exit(1)
        elif not os.path.exists(args.template):
            print(f"Error: Template file '{args.template}' not found.")
            exit(1)

        # Process
        if args.order is None:
            inspect_all_orders(
                args.file, args.template, args.tapas,
                instrument=args.instrument,
                show_oh=not args.no_oh, show_tapas=not args.no_tapas,
                output_dir=args.output_dir
            )
        elif len(args.order) == 1:
            inspect_single_order(
                args.file, args.template, args.order[0],
                tapas_file=args.tapas, instrument=args.instrument,
                show_oh=not args.no_oh, show_tapas=not args.no_tapas,
                save_pdf=not args.no_save, show=not args.no_show,
                output_dir=args.output_dir
            )
        else:
            inspect_all_orders(
                args.file, args.template, args.tapas,
                instrument=args.instrument,
                order_min=args.order[0], order_max=args.order[1],
                show_oh=not args.no_oh, show_tapas=not args.no_tapas,
                output_dir=args.output_dir
            )
    else:
        parser.print_help()
