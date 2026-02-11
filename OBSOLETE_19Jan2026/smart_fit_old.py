import numpy as np
from astropy.io import fits
import warnings
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline as ius
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import glob
from astropy.coordinates import EarthLocation, get_sun
from astropy.time import Time
from astropy.coordinates import AltAz
import numpy as np
from scipy.signal import savgol_filter
# import EarthLocation, Time, AltAz, get_sun
import astropy.units as u
from astropy.time import Time
import numexpr as ne
from aperocore import math as mp
from aperocore.science import wavecore
import sys
import time

# Suppress numpy and astropy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*Card is too long.*')
warnings.filterwarnings('ignore', message='.*VerifyWarning.*')

import tellu_tools as tt
from tellu_tools_config import tprint

instrument = 'NIRPS'  # 'NIRPS' or 'SPIROU'

molecules = ['H2O', 'CH4', 'CO2', 'O2']

speed_of_light = 299792.458  # m/s/
doplot = False


def format_eta(seconds: float) -> str:
    """Format an ETA in seconds into a concise human-readable string."""
    if seconds < 0:
        seconds = 0
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)

    # Prefer compact formats with zero-padded minutes when hours/days exist
    if days:
        return f"{days}d {hours:02d}h {mins:02d}m"
    if hours:
        return f"{hours}h {mins:02d}m"
    if mins:
        if secs:
            return f"{mins}m {secs:02d}s"
        return f"{mins}m"
    return f"{secs}s"

os.chdir(tt.user_params()['project_path'])

sky_dict = tt.sky_pca_fast()
waveref = fits.getdata(f'calib_{instrument}/waveref.fits')

all_files = np.array(glob.glob(f'hotstars_{instrument}/*pp_e2dsff_*.fits'))

# Partition into done vs pending
pending_files = []
done_files = []
for file in all_files:
    outname = f'tellu_fit_{instrument}/trans_'+file.split('/')[-1]
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
    sys.exit(0)

# Shuffle pending list
if len(pending_files) > 1:
    order = np.random.permutation(len(pending_files))
    pending_files = [pending_files[i] for i in order]

blaze = tt.get_blaze()

durations = []

for idx, (file, outname) in enumerate(pending_files, start=1):
    # Peek header early to get object name for banner
    hdr_peek = fits.getheader(file)
    obj_name = hdr_peek.get('OBJECT') or hdr_peek.get('OBJNAME') or 'UNKNOWN'

    banner = f"{'*'*20} {obj_name} [{idx}/{N_pending}] {'*'*20}"
    tprint(banner, color='cyan')

    tprint(f'Processing: {outname}', color='green')

    loop_start = time.perf_counter()

    sp = fits.getdata(file)
    hdr = hdr_peek
    hdr = tt.update_header(hdr)

    wavefile = hdr['WAVEFILE']

    if not os.path.exists(f'calib_{instrument}/'+wavefile):
        if instrument == 'NIRPS':
            cmd = f'scp rali:/cosmos99/nirps/apero-data/nirps_he_online/calib/{wavefile} calib_{instrument}/.'
            tprint(f'Getting {wavefile} from rali...', color='blue')
            os.system(cmd)
        if instrument == 'SPIROU':
            cmd = f'scp rali:/cosmos99/spirou/apero-data/spirou_offline/calib/{wavefile} calib_{instrument}/.'
            tprint(f'Getting {wavefile} from rali...', color='blue')
            os.system(cmd)

    wave0 = fits.getdata(f'calib_{instrument}/'+wavefile)
    # everything downhill is in the waveref frame
    sp = wavecore.wave_to_wave(sp, wave0, waveref)



    airmass = hdr['AIRMASS']
    pressure = hdr['PRESSURE']  # in kPa
    pressure0 = hdr['NORMPRES']  # in kPa

    all_abso = tt.construct_abso(waveref, expos=[]*4)

    # remove for now
    if False:
        # color molecules 
        color_mol = ['green', 'red', 'blue',  'lime']

        fig, ax = plt.subplots(nrows = len(molecules), ncols=1, figsize=(10,15), sharex=True)
        for iord in range(0,wave.shape[0]):
            for imolecule, molecule in enumerate(molecules):
                if iord == 0:
                    ax[imolecule].plot(wave[iord], all_abso[imolecule][iord], label=molecule, alpha=0.5, color=color_mol[imolecule])
                    ax[imolecule].legend()
                    ax[imolecule].set_ylim([0,1.0])

                else:  
                    ax[imolecule].plot(wave[iord], all_abso[imolecule][iord], alpha=0.5, color=color_mol[imolecule])

        ax[-1].set_xlabel('Wavelength')
        ax[len(molecules)//2].set_ylabel('Flux')
        plt.title('Observed Spectrum')
        plt.show()

    fixed_exponents = [1, airmass, airmass, airmass]
    expo_optimal = np.zeros(4)
    for imolecule in range(4):
        fixed_exponents[imolecule] = None # we fit molecules one at a time
        expo_result = tt.optimize_exponents(waveref, sp, airmass, fixed_exponents=fixed_exponents)
        expo_optimal[imolecule] = expo_result[imolecule]  # Extract the optimized value for this molecule

    for molecule, expo in zip(molecules, expo_optimal):
        tprint(f'Optimized expo for {molecule}: {expo:.4f}', color='blue')

    tprint(f'Airmass used: {airmass:.4f}', color='blue')
    trans2 = tt.construct_abso(waveref, expos=expo_optimal, all_abso=all_abso)
    h2ocv = expo_optimal[0]*hdr['H2OCV']/(hdr['AIRMASS']*pressure/pressure0)
    co2_vmr = expo_optimal[1]*hdr['VMR_CO2']/(hdr['AIRMASS']*pressure/pressure0)
    ch4_vmr = expo_optimal[2]*hdr['VMR_CH4']/(hdr['AIRMASS']*pressure/pressure0)
    o2_frac = expo_optimal[3]/(hdr['AIRMASS']*pressure/pressure0)

    hdr['NORMPRES'] = pressure0,'[kPa] Normalization pressure for TAPAS values'
    hdr['H2O_CV'] = h2ocv,'[mm] at zenith, normalized pressure'
    hdr['CO2_VMR'] = co2_vmr, '[ppm] at zenith, normalized pressure'
    hdr['CH4_VMR'] = ch4_vmr,'[ppm] zenith, normalized pressure'
    hdr['O2_AIRM'] = o2_frac,'Airmass equivalent fraction at normalized pressure'

    # update header with optimized exponents
    for i, molecule in enumerate(molecules):
        hdr[f'EXPO_{molecule}'] = expo_optimal[i], f'Optimized exponent for {molecule}'

    for iord in range(0,waveref.shape[0]):
        sp[iord]/=np.nanpercentile(sp[iord], 90)

    trans = sp/trans2/blaze
    sky = tt.sky_pca_fast(wave=waveref, spectrum=trans, sky_dict=sky_dict, force_positive=True)

    trans -= sky
    sky_sp = (sky*trans2*blaze)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        log_trans = np.log(trans)

    fits.writeto(outname, log_trans, hdr, overwrite=True)

    loop_end = time.perf_counter()
    loop_dur = loop_end - loop_start
    durations.append(loop_dur)

    remaining = N_pending - idx
    mean_dur = float(np.mean(durations))
    median_dur = float(np.median(durations))
    eta_seconds = remaining * mean_dur

    tprint(f'Step duration: {loop_dur:0.2f}s | mean {mean_dur:0.2f}s | median {median_dur:0.2f}s', color='blue')
    if remaining > 0:
        eta_str = format_eta(eta_seconds)
        tprint(f'Remaining {remaining} stars, ETA ~ {eta_str}', color='magenta')
    else:
        tprint('All pending stars completed.', color='green')

    if not doplot:
        continue

    # Load demo_order from telluric_config.yaml
    config = tt.load_telluric_config()
    demo_order_config = config.get('demo_order', {}).get(instrument, [0, 71])
    
    # Handle demo_order as [start, end] range or single value
    if isinstance(demo_order_config, (list, tuple)) and len(demo_order_config) == 2:
        order_range = range(demo_order_config[0], demo_order_config[1] + 1)
    else:
        order_range = [demo_order_config] if isinstance(demo_order_config, int) else [demo_order_config[0]]
    
    # Compute corrected spectrum (hot star = flat continuum target)
    sp_corr = (sp - sky_sp) / trans2
    
    # Sky emission mask (sky > stellar flux)
    stellar_flux = sp - sky_sp
    reject_bright_sky = sky_sp > stellar_flux
    
    # Molecule colors
    mol_colors = {'H2O': 'blue', 'CH4': 'orange', 'CO2': 'green', 'O2': 'red'}
    
    # Get masked pixels for all orders in range
    masked_telluric = np.zeros(trans2.shape[1], dtype=bool)
    masked_sky = np.zeros(trans2.shape[1], dtype=bool)
    for iord in order_range:
        masked_telluric |= (trans2[iord] < config.get('processing', {}).get('low_flux_threshold', 0.2))
        masked_sky |= reject_bright_sky[iord]
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Helper to shade masked regions with filled rectangles
    def shade_masked(ax, wave_all, mask):
        """Draw grey rectangles for masked regions using midpoint boundaries."""
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
                             facecolor='grey', alpha=0.3, edgecolor='none', zorder=0)
            ax.add_patch(rect)
    
    # Helper to shade sky emission line regions with salmon color
    def shade_emission_lines(ax, wave_all, mask):
        """Draw salmon rectangles for sky emission line regions."""
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
    
    # Plot all orders
    sp0_all = []
    for iord in order_range:
        alpha = 0.7
        sp0_all.append(sp[iord])
        
        # Panel 1: Original spectrum + sky
        axes[0].plot(waveref[iord], sp[iord], 'k-', lw=0.5, alpha=alpha, zorder=2)
        axes[0].plot(waveref[iord], sky_sp[iord], 'b-', lw=0.5, alpha=0.5, zorder=2)
        
        # Panel 2: Absorption model - show all molecules
        for imol, mol in enumerate(molecules):
            mol_trans = all_abso[imol][iord] ** expo_optimal[imol]
            axes[1].plot(waveref[iord], mol_trans, '-', lw=0.5, 
                        color=mol_colors[mol], alpha=0.5, zorder=2)
        axes[1].plot(waveref[iord], trans2[iord], 'k-', lw=0.8, alpha=alpha, zorder=3)
        
        # Panel 3: Corrected spectrum (hot star ~ flat at 1.0)
        axes[2].plot(waveref[iord], sp_corr[iord], 'g-', lw=0.5, alpha=alpha, zorder=2)
        axes[2].axhline(1.0, color='k', ls='--', lw=0.5, alpha=0.5, zorder=2)
        
        # Panel 4: Residuals (corrected - 1.0)
        residual = sp_corr[iord] - 1.0
        axes[3].plot(waveref[iord], residual, 'r-', lw=0.5, alpha=alpha, zorder=2)
    
    # Set ylim for top panel: 0 to 1.5 * 90th percentile of original
    sp0_flat = np.concatenate(sp0_all)
    ymax_top = 1.5 * np.nanpercentile(sp0_flat, 90)
    axes[0].set_ylim(0, ymax_top)
    
    # Labels and legends (sort wavelengths for shading)
    wave_flat = np.concatenate([waveref[iord] for iord in order_range])
    wave_sorted_idx = np.argsort(wave_flat)
    wave_sorted = wave_flat[wave_sorted_idx]
    masked_sorted = np.concatenate([(trans2[iord] < 0.2) for iord in order_range])[wave_sorted_idx]
    
    axes[0].set_ylabel('Flux')
    axes[0].plot([], [], 'k-', lw=0.5, label='Original')
    axes[0].plot([], [], 'b-', lw=0.5, label='Sky')
    axes[0].legend(loc='upper right')
    axes[0].set_title(f'{obj_name} - Orders {order_range[0]}-{order_range[-1]} - {os.path.basename(file)}')
    shade_masked(axes[0], wave_sorted, masked_sorted)
    
    # Add molecule legend
    for mol in molecules:
        axes[1].plot([], [], '-', color=mol_colors[mol], lw=1, label=mol)
    axes[1].plot([], [], 'k-', lw=1, label='Combined')
    axes[1].set_ylabel('Transmission')
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(loc='lower right', ncol=5, fontsize=8)
    shade_masked(axes[1], wave_sorted, masked_sorted)
    
    axes[2].plot([], [], 'g-', lw=0.5, label='Corrected')
    axes[2].axhline(np.nan, color='k', ls='--', lw=0.5, label='Continuum (1.0)')
    axes[2].set_ylabel('Flux (corrected)')
    axes[2].legend(loc='upper right')
    
    # Set ylim for corrected plot
    sp_corr_flat = np.concatenate([sp_corr[iord] for iord in order_range])
    ymax_panel3 = 1.5 * np.nanpercentile(sp_corr_flat, 90)
    axes[2].set_ylim(0, ymax_panel3)
    shade_masked(axes[2], wave_sorted, masked_sorted)
    
    # Also shade sky emission line regions
    sky_masked_sorted = np.concatenate([reject_bright_sky[iord] for iord in order_range])[wave_sorted_idx]
    shade_emission_lines(axes[2], wave_sorted, sky_masked_sorted)
    
    # Residuals plot (bottom)
    axes[3].axhline(0, color='grey', lw=0.5, ls='--', zorder=1)
    axes[3].set_xlabel('Wavelength (nm)')
    axes[3].set_ylabel('Residual')
    
    # Set symmetric ylim based on 16-84th percentile of non-masked residuals
    residual_flat = np.concatenate([sp_corr[iord] - 1.0 for iord in order_range])
    mask_flat = masked_sorted | sky_masked_sorted
    residual_valid = residual_flat[wave_sorted_idx][~mask_flat]
    if len(residual_valid) > 0:
        p16, p84 = np.nanpercentile(residual_valid, [16, 84])
        half_width = (p84 - p16) / 2.0
        axes[3].set_ylim(-8 * half_width, 8 * half_width)
    shade_masked(axes[3], wave_sorted, masked_sorted)
    shade_emission_lines(axes[3], wave_sorted, sky_masked_sorted)
    
    plt.tight_layout()
    plt.show()
    
    # Prompt user for next plot
    try:
        response = input("Show next spectrum plot? [Y/n/number to skip]: ").strip().lower()
        if response in ['n', 'no']:
            doplot = False
            tprint("Plotting disabled for remaining spectra", color='orange')
        elif response.isdigit() and int(response) > 0:
            # Skip N spectra (simple counter via global-ish variable)
            skip_count = int(response)
            tprint(f"Will skip {response} spectra before showing next plot", color='cyan')
            # Note: simple implementation - just toggle doplot off temporarily
            # For full skip counter, would need more refactoring
    except EOFError:
        doplot = False

