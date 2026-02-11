from astropy.io import fits
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
# import curve fit
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

# Paper figure tracking
_paper_figure_done = {'fig7': False}


def get_paper_figures_config(instrument: str = 'NIRPS'):
    """Get paper figures configuration from yaml."""
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

def anthropic(mjds_airmass,*params):
    # params: 
    # slope
    # intercept
    # amp
    # yearly phase
    # airmass exponent
    # temperature exponent
    # pressure exponent

    mjds = mjds_airmass[0]
    airmass = mjds_airmass[1]
    temperature = mjds_airmass[2]
    pressure_norm = mjds_airmass[3]  
    y = (params[0]*mjds + params[1]  + params[2]*np.cos(2.0*np.pi*(mjds%365.24)/365.24 + params[3]))
    return y*airmass**params[4]*(temperature)**params[5]*(pressure_norm)**params[6]

def simple_scaling(airmass_temperature_pressure,*params):
    # airmass exponent
    # temperature exponent
    # pressure exponent

    airmass = airmass_temperature_pressure[0]
    temperature = airmass_temperature_pressure[1]
    pressure_norm = airmass_temperature_pressure[2]  

    return airmass**params[0]*(temperature)**params[1]*(pressure_norm)**params[2]+params[3]

def accurate_airmass(z):
    R_earth = 6371.0  # km
    H_atmo = 8.43     # km
    z_rad = np.radians(z)
    sec_z = 1.0/np.cos(z_rad)
    am = np.sqrt((R_earth/(R_earth + H_atmo))**2 * sec_z**2 - (R_earth/(R_earth + H_atmo))**2 + 1.0)
    return am

tbl_params_fit = dict()

#instrument = 'SPIROU'
instrument = 'NIRPS'

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

# Get list of telluric fit files
files = glob.glob(os.path.join(project_path, f'tellu_fit_{instrument}/trans_*.fits'))
files = sorted(files)

if not os.path.exists(big_table_file):

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
    tbl = Table.read(big_table_file, format='csv')
    keys = [col for col in tbl.colnames if col != 'FILENAME']

tbl['MORNING'] = tbl['SUNSETD']>5

# Get valid hot stars list from telluric config
telluric_config = load_telluric_config()
valid_hot_stars = telluric_config.get('hot_stars', [])

# Mark hot stars based on config list (not header)
tbl['HOTSTAR'] = np.array([obj in valid_hot_stars for obj in tbl['DRSOBJN']])

not_hot_stars = np.unique(tbl[tbl['HOTSTAR'] == False]['DRSOBJN'])

for obj in not_hot_stars:
    tprint(f'The object {obj} is not in hot_stars list, removing from the sample', color='blue')
    bad = (tbl['DRSOBJN'] == obj)
    tbl = tbl[~bad]

    
# for all columns, try to convert to float, otherwise it's a string
for key in keys:

    if tbl[key][0] in [True,False]:
        tbl[key] = tbl[key].astype(bool)
        continue

    try:
        tbl[key] = tbl[key].astype(float)
    except:
        pass


# Load O2_AIRM thresholds from config
config = load_telluric_config()
qc_config = config.get('quality_control', {})
o2_airm_min = qc_config.get('o2_airm_min', 0.92)
o2_airm_max = qc_config.get('o2_airm_max', 1.08)

# Mark O2_AIRM validity (fluorescence contamination check)
# Only affects O2 fitting - other molecules use all data
tbl['O2_VALID'] = (tbl['O2_AIRM'] >= o2_airm_min) & (tbl['O2_AIRM'] <= o2_airm_max)
n_o2_valid = np.sum(tbl['O2_VALID'])
n_o2_excluded = len(tbl) - n_o2_valid
tprint(f'O2_AIRM validity: {n_o2_valid} valid, {n_o2_excluded} excluded (range: {o2_airm_min}-{o2_airm_max})', color='blue')

bad = tbl['AIRMASS']>2.0
tbl = tbl[~bad]
bad = tbl['SNR_REF'] <100
tbl = tbl[~bad]


sigmax = np.inf

tbl['MORNING'] = tbl['SUNSETD'] > 5
tbl['EVENING'] = tbl['SUNSETD'] < 5


#tbl['AIRMASS'] = tbl['ACCAIRM']
tbl['NORMALIZED_PRESSURE'] = tbl['NORMPRES']/tbl['PRESSURE']
tbl['NORMALIZED_TEMPERAT'] = tbl['TEMPERAT']/273.15


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
global _paper_figure_done
enabled, output_dir = get_paper_figures_config(instrument)
if enabled and not _paper_figure_done['fig7']:
    fig_path = os.path.join(output_dir, 'fig7_qc_diagnostics.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    tprint(f'Paper figure saved: {fig_path}', color='green')
    _paper_figure_done['fig7'] = True

plt.savefig('o2_airmass_hotstar_check.png')
plt.show()

tbl = tbl[tbl['HOTSTAR']]

# O2 fit uses ONLY morning (>midnight) observations with valid O2_AIRM
# Early night observations are contaminated by O2 fluorescence (Meinel bands)
# excited by UV during twilight. By midnight, the fluorescence has decayed.
# O2_VALID mask excludes fluorescence-contaminated spectra
o2_fit_mask = tbl['MORNING'] & tbl['O2_VALID']
while sigmax > sigma_cut:

    # construct a 2 parameter linear fit, one with AIRMASS, the other one with temperature
    vv = [tbl['AIRMASS'], tbl['NORMALIZED_TEMPERAT'], tbl['NORMALIZED_PRESSURE']]
    curve_fit_o2, curve_fit_cov = curve_fit(simple_scaling, [vv[0][o2_fit_mask], vv[1][o2_fit_mask], vv[2][o2_fit_mask]], tbl['O2_AIRM'][o2_fit_mask], p0=[1.0, 1.0, 1.0, 0.0])


    residual = tbl['O2_AIRM'][o2_fit_mask] - simple_scaling([vv[0][o2_fit_mask], vv[1][o2_fit_mask], vv[2][o2_fit_mask]], *curve_fit_o2)
    sigmax = np.max(np.abs(residual/mp.robust_nanstd(residual)))
    if sigmax > sigma_cut:
        tprint(f'Removing O2 outlier with {sigmax:.2f} sigma max deviation', color='yellow')
        maxid = np.argmax(np.abs(residual/mp.robust_nanstd(residual)))
        # Mark as invalid in O2_VALID instead of removing the row
        o2_fit_indices = np.where(o2_fit_mask)[0]
        tbl['O2_VALID'][o2_fit_indices[maxid]] = False
        o2_fit_mask = tbl['MORNING'] & tbl['O2_VALID']

tbl_params_fit['O2_AIRMASS_EXP'] = curve_fit_o2[0]
tbl_params_fit['O2_TEMPERATURE_EXP'] = curve_fit_o2[1]
tbl_params_fit['O2_PRESSURE_EXP'] = curve_fit_o2[2]
tbl_params_fit['O2_INTERCEPT'] = curve_fit_o2[3]
                                                        
tprint('O2 airmass fit parameters (morning):', color='blue')
tprint(f'Airmass exponent: {curve_fit_o2[0]:.6f}', color='blue')
tprint(f'Temperature exponent: {curve_fit_o2[1]:.6f}', color='blue')
tprint(f'Pressure exponent: {curve_fit_o2[2]:.6f}', color='blue')
tprint(f'Intercept: {curve_fit_o2[3]:.6f}', color='blue')


plt.figure()
plt.plot(tbl['NORMALIZED_TEMPERAT'][o2_fit_mask], residual, 'ro')
plt.xlabel('Temperature [K]')
plt.ylabel('O2 airmass fit residuals (morning)')
plt.grid()
plt.show()

# same with pressure
plt.figure()
plt.plot(tbl['NORMALIZED_PRESSURE'][o2_fit_mask], residual, 'ro')
plt.xlabel('Pressure [kPa]')
plt.ylabel('O2 airmass fit residuals (morning)')
plt.grid()
plt.show()

tprint(f'O2 airmass fit residuals std (morning): {mp.robust_nanstd(residual*100):.2f}%', color='blue')


plt.figure()
plt.scatter(tbl['AIRMASS'][tbl['EVENING']], tbl['O2_AIRM'][tbl['EVENING']], c=tbl['DRSSUNEL'][tbl['EVENING']], cmap='viridis', alpha=0.3)
plt.plot(tbl['AIRMASS'][o2_fit_mask], tbl['O2_AIRM'][o2_fit_mask], 'ro', label='Morning observations (valid O2)', alpha=0.5)
plt.colorbar(label='Sun angle below horizon [hours]')
plt.xlabel('AIRMASS')
plt.ylabel('O2_AIRM')
plt.show()


nsigmax = np.inf

while nsigmax > sigma_cut:
    mjds_airmass = [tbl['MJDMID'], tbl['AIRMASS'], tbl['NORMALIZED_TEMPERAT'],tbl['NORMALIZED_PRESSURE']]
    curve_fit_co2, curve_fit_cov = curve_fit(anthropic, mjds_airmass, tbl['CO2_VMR'], p0=[0.0, 400.0, 50.0, 0.0, 1.0,1.0,1.0])
    # residual to fit
    curve_fit_co2_zenith = curve_fit_co2.copy()
    curve_fit_co2_zenith[4] = 0.0  # airmass exponent set to 0 for zenith
    curve_fit_co2_zenith[5] = 0.0  # temperature exponent set to 0 for reference temp
    curve_fit_co2_zenith[6] = 0.0  # pressure exponent set to 0 for reference pressure

    tbl['RESIDUAL_CO2'] = tbl['CO2_VMR']/tbl['AIRMASS']**curve_fit_co2[4]/(tbl['NORMALIZED_TEMPERAT'])**curve_fit_co2[5]/tbl['NORMALIZED_PRESSURE']**curve_fit_co2[6] - anthropic([tbl['MJDMID'], np.ones_like(tbl['MJDMID']), 273.15*np.ones_like(tbl['MJDMID']), np.ones_like(tbl['MJDMID'])], *curve_fit_co2_zenith)

    nsigmax = np.max(np.abs(tbl['RESIDUAL_CO2']/mp.robust_nanstd(tbl['RESIDUAL_CO2'])))
    if nsigmax > sigma_cut:
        print(f'Removing outlier with {nsigmax:.2f} sigma max deviation')
        maxid = np.argmax(np.abs(tbl['RESIDUAL_CO2']/mp.robust_nanstd(tbl['RESIDUAL_CO2'])))
        tbl.remove_row(maxid)


tbl_params_fit['CO2_SLOPE'] = curve_fit_co2[0]
tbl_params_fit['CO2_INTERCEPT'] = curve_fit_co2[1]
tbl_params_fit['CO2_AMP'] = curve_fit_co2[2]
tbl_params_fit['CO2_PHASE'] = curve_fit_co2[3]
tbl_params_fit['CO2_AIRMASS_EXP'] = curve_fit_co2[4]
tbl_params_fit['CO2_TEMPERATURE_EXP'] = curve_fit_co2[5]
tbl_params_fit['CO2_PRESSURE_EXP'] = curve_fit_co2[6]

# print fit parameters
tprint('CO2 fit parameters:', color='blue')
tprint(f'Slope: {curve_fit_co2[0]*365.24:.2f} ppm/year', color='blue')
tprint(f'Intercept: {curve_fit_co2[1]:.6f} ppm', color='blue')
tprint(f'Sinusoidal amplitude: {curve_fit_co2[2]:.6f} ppm', color='blue')
tprint(f'Sinusoidal phase: {curve_fit_co2[3]:.6f} rad', color='blue')
tprint(f'Airmass exponent: {curve_fit_co2[4]:.6f}', color='blue')
tprint(f'Temperature exponent: {curve_fit_co2[5]:.6f}', color='blue')
tprint(f'Pressure exponent: {curve_fit_co2[6]:.6f}', color='blue')

# plot the fit with the sinusoidal variation but not the airmass dependence. use airmass = 1.0 and temp = 273.15 K
mjds_fit = np.linspace(np.min(tbl['MJDMID']), np.max(tbl['MJDMID']), 1000)

co2_fit = anthropic([mjds_fit, np.ones_like(mjds_fit), np.ones_like(mjds_fit), np.ones_like(mjds_fit)], *curve_fit_co2_zenith)

plt.figure()
plt.plot(mjds_fit, co2_fit, 'r-', label='Fit',linewidth=2)
plt.scatter(tbl['MJDMID'],tbl['CO2_VMR']/tbl['AIRMASS']**curve_fit_co2[4]/(tbl['NORMALIZED_TEMPERAT'])**curve_fit_co2[5]/tbl['NORMALIZED_PRESSURE']**curve_fit_co2[6], c=tbl['AIRMASS'], cmap='viridis', label='CO2 VMR')
#plt.plot(tbl['MJDMID'][morning],tbl['CO2_VMR'][morning], 'ro', label='Morning observations')
plt.xlabel('MJDMID')
plt.ylabel('CO2 VMR [ppm]')
# add legend
plt.legend()
plt.colorbar(label='Airmass')
plt.show()


tprint(f'Fractional CO2 fit residuals : {mp.robust_nanstd(tbl["RESIDUAL_CO2"])/np.nanmedian(tbl["CO2_VMR"])*100:.2f}%', color='blue')

nsigmax = np.inf

while nsigmax > sigma_cut:
    mjds_airmass = [tbl['MJDMID'], tbl['AIRMASS'], tbl['NORMALIZED_TEMPERAT'], tbl['NORMALIZED_PRESSURE']]
    curve_fit_ch4, curve_fit_cov = curve_fit(anthropic, mjds_airmass, tbl['CH4_VMR'], p0=[0.0, 1500.0, 200.0, 0.0, 1.0,1.0,1.0])

    curve_fit_ch4_zenith = curve_fit_ch4.copy()
    curve_fit_ch4_zenith[4] = 0.0  # airmass exponent set to 0 for zenith
    curve_fit_ch4_zenith[5] = 0.0  # temperature exponent set to 0 for reference temp
    curve_fit_ch4_zenith[6] = 0.0  # pressure exponent set to 0 for reference pressure


    tbl['RESIDUAL_CH4'] = tbl['CH4_VMR']/tbl['AIRMASS']**curve_fit_ch4[4]/(tbl['NORMALIZED_TEMPERAT'])**curve_fit_ch4[5]/tbl['NORMALIZED_PRESSURE']**curve_fit_ch4[6] - anthropic([tbl['MJDMID'], np.ones_like(tbl['MJDMID']), 273.15*np.ones_like(tbl['MJDMID']), np.ones_like(tbl['MJDMID'])], *curve_fit_ch4_zenith)

    nsigmax = np.max(np.abs(tbl['RESIDUAL_CH4']/mp.robust_nanstd(tbl['RESIDUAL_CH4'])))
    if nsigmax > sigma_cut:
        print(f'Removing outlier with {nsigmax:.2f} sigma max deviation')
        maxid = np.argmax(np.abs(tbl['RESIDUAL_CH4']/mp.robust_nanstd(tbl['RESIDUAL_CH4'])))
        tbl.remove_row(maxid)


mjds_fit = np.linspace(np.min(tbl['MJDMID']), np.max(tbl['MJDMID']), 1000)

tprint('CH4 fit parameters:', color='blue')
tprint(f'Slope: {curve_fit_ch4[0]*365.24:.2f} ppm/year', color='blue')
tprint(f'Intercept: {curve_fit_ch4[1]:.6f} ppm', color='blue')
tprint(f'Sinusoidal amplitude: {curve_fit_ch4[2]:.6f} ppm', color='blue')
tprint(f'Sinusoidal phase: {curve_fit_ch4[3]:.6f} rad', color='blue')
tprint(f'Airmass exponent: {curve_fit_ch4[4]:.6f}', color='blue')
tprint(f'Temperature exponent: {curve_fit_ch4[5]:.6f}', color='blue')
tprint(f'Pressure exponent: {curve_fit_ch4[6]:.6f}', color='blue')

tbl_params_fit['CH4_SLOPE'] = curve_fit_ch4[0]
tbl_params_fit['CH4_INTERCEPT'] = curve_fit_ch4[1]
tbl_params_fit['CH4_AMP'] = curve_fit_ch4[2]
tbl_params_fit['CH4_PHASE'] = curve_fit_ch4[3]
tbl_params_fit['CH4_AIRMASS_EXP'] = curve_fit_ch4[4]
tbl_params_fit['CH4_TEMPERATURE_EXP'] = curve_fit_ch4[5]
tbl_params_fit['CH4_PRESSURE_EXP'] = curve_fit_ch4[6]

ch4_fit = anthropic([mjds_fit, np.ones_like(mjds_fit), 273.15*np.ones_like(mjds_fit), np.ones_like(mjds_fit)], *curve_fit_ch4_zenith)
plt.figure()
plt.plot(mjds_fit, ch4_fit, 'r-', label='Fit', linewidth=2)

print( np.nanstd(tbl['RESIDUAL_CH4'])/np.nanmedian(tbl['CH4_VMR']))

plt.scatter(tbl['MJDMID'],tbl['CH4_VMR']/tbl['AIRMASS']**curve_fit_ch4[4]/(tbl['NORMALIZED_TEMPERAT'])**curve_fit_ch4[5]/tbl['NORMALIZED_PRESSURE']**curve_fit_ch4[6], c=tbl['AIRMASS'], cmap='viridis', label='CH4 VMR')
plt.xlabel('MJDMID')
plt.ylabel('CH4 VMR [ppm]')
plt.colorbar(label='Airmass')
plt.show()

print(f'Fractional CH4 fit residuals : {mp.robust_nanstd(tbl["RESIDUAL_CH4"])/np.nanmedian(tbl["CH4_VMR"])*100:.2f}%')

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

tbl_params.write(outname, format='csv', overwrite=True)


mean_expos = [np.nanmedian(tbl['EXPO_H2O']/tbl['AIRMASS']),np.nanmedian(tbl['EXPO_CO2']/tbl['AIRMASS']),np.nanmedian(tbl['EXPO_CH4']/tbl['AIRMASS']),np.nanmedian(tbl['EXPO_O2']/tbl['AIRMASS'])]
tprint('Mean exponents at zenith:', color='green')
tprint(f'H2O: {mean_expos[0]:.3f}, CO2: {mean_expos[1]:.3f}, CH4: {mean_expos[2]:.3f}, O2: {mean_expos[3]:.3f}', color='blue')

hdr = fits.getheader(files[0])

all_abso = construct_abso(waveref, mean_expos, all_abso=None)
mean_abso = np.product(all_abso, axis=0)
ceil_abso = np.ones_like(all_abso[0]) * 0.95

# Append ceil_abso as the last slice
all_abso = np.concatenate([all_abso, ceil_abso[None, :, :]], axis=0)

# Find the main absorber at each wavelength
main_absorber = np.argmin(all_abso, axis=0)

outname = 'main_absorber_'+instrument+'.fits'
fits.writeto(outname, main_absorber.astype(np.int16), overwrite=True)

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