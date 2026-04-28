from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import glob
from astropy.io import fits
from tqdm import tqdm

"""
tbl = Table.read('lbl_TOI4552pca_TOI4552pca.rdb')
#tbl = Table.read('lbl2_TOI4552batch1_TOI4552batch1.rdb')

keys  = tbl.keys()
# find keys starting with vrad and having nm in them
vrad_keys = [key for key in keys if key.startswith('vrad') and 'nm' in key]
# get the nm for each considering the syntax vrad_1170nm
nms = [int(key.split('_')[1].replace('nm','')) for key in vrad_keys]

# color-code from blue (low nm) to red (high nm)
colors = plt.cm.inferno(np.linspace(0,1,len(nms)))
for i, key in enumerate(vrad_keys):
    # if 0 in tbl[key] then skip
    if np.all(tbl[key] == 0):
        continue

    # error bars are s[key] if exists
    plt.errorbar(tbl['BERV'] + 26.000, tbl[key], yerr=tbl['s' + key], fmt='o', color=colors[i], label=f'{nms[i]} nm', alpha=0.2)


plt.errorbar(tbl['BERV'] + 26.000, tbl['vrad'], yerr=tbl['svrad'], fmt='o', color='black', label='Overall vrad', alpha=0.5)
plt.errorbar(tbl['BERV'] + 26.000, tbl['vrad_h'], yerr=tbl['svrad_h'], fmt='o', color='blue', label='Overall H vrad', alpha=0.5)

plt.xlabel('BERV (km/s)')
plt.ylabel('Radial Velocity (km/s)')
plt.title('Radial Velocity vs BERV for Different Wavelengths')
plt.legend(title='Wavelength (nm)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()


plt.tight_layout()
plt.show()
"""


files = glob.glob('/Users/eartigau/lbl_NIRPS/lblrv/TOI4552pca_TOI4552pca/*.fits')



# side 1 berv from -28 to -26
# side 2 berv from -26 to -24
# else skip

n_side_1 = 0
n_side_2 = 0


tbl0 = Table.read(files[0])
tbl_wave_start = tbl0['WAVE_START']
XPIX = tbl0['XPIX']
dv_1 = np.zeros(len(tbl_wave_start))
sdv_1 = np.zeros(len(tbl_wave_start))

dv_2 = np.zeros(len(tbl_wave_start))
sdv_2 = np.zeros(len(tbl_wave_start))

for file in tqdm(files, desc='Processing files',leave=False):
    tbl = Table.read(file)
    hdr = fits.getheader(file)
    berv = hdr['BERV']
    if -28.0 <= berv <= -26.0:
        n_side_1 += 1
        dv_1 += tbl['dv']
        sdv_1 += tbl['sdv']**2
    elif -26.0 < berv <= -24.0:
        n_side_2 += 1
        dv_2 += tbl['dv']
        sdv_2 += tbl['sdv']**2
    else:
        continue

print(f'Side 1 count: {n_side_1}, Side 2 count: {n_side_2}')

dv_1 /= n_side_1
sdv_1 = np.sqrt(sdv_1) / n_side_1

dv_2 /= n_side_2
sdv_2 = np.sqrt(sdv_2) / n_side_2

#keep = (sdv_1>200) & (sdv_2>200)
cut_level = 1e9
keep = (sdv_1<cut_level) & (sdv_2<cut_level)
tbl_wave_start = tbl_wave_start[keep]
dv_1 = dv_1[keep]
sdv_1 = sdv_1[keep]
dv_2 = dv_2[keep]
sdv_2 = sdv_2[keep]
XPIX = XPIX[keep]



roi = (tbl_wave_start > 1000) & (tbl_wave_start < 1150)
plt.errorbar(sdv_1[roi],(sdv_1- sdv_2)[roi], yerr = np.sqrt(sdv_1[roi]**2 + sdv_2[roi]**2), fmt='k.', alpha=0.2, zorder =1)
plt.xlabel('Side 1 sdv (km/s)')
plt.ylabel('Side 1 sdv - Side 2 sdv (km/s)')
plt.title('Difference in sdv between Side 1 and Side 2 vs Side 1 sdv')
plt.plot(sdv_1[roi],(sdv_1- sdv_2)[roi], 'r.', zorder =10)
plt.show()

# plot as a function of XPIX wavelength with error bars and color-code by wavelength

# find the weighted-mean dv_1 and dv_2 for XPIX between 500 and 1000
valid_xpix = (XPIX > 2500) & (XPIX < 3500) &  (tbl_wave_start > 1000) & (tbl_wave_start < 1250)
mean_dv_1 = np.nansum(dv_1[valid_xpix]/sdv_1[valid_xpix]**2)/np.nansum(1/sdv_1[valid_xpix]**2)
err_dv_1 = 1/np.sqrt(np.sum(1/sdv_1[valid_xpix]**2))
mean_dv_2 = np.nansum(dv_2[valid_xpix]/sdv_2[valid_xpix]**2)/np.nansum(1/sdv_2[valid_xpix]**2)
err_dv_2 = 1/np.sqrt(np.sum(1/sdv_2[valid_xpix]**2))
print('Weighted Mean Side 1 dv:', mean_dv_1, '±', err_dv_1)
print('Weighted Mean Side 2 dv:', mean_dv_2, '±', err_dv_2)

good_wave = (tbl_wave_start > 1000) & (tbl_wave_start < 1250)
plt.errorbar(XPIX[good_wave], dv_1[good_wave], yerr=sdv_1[good_wave], fmt='o', label='Side 1 BERV [-28,-26] km/s', alpha=0.2)
plt.errorbar(XPIX[good_wave], dv_2[good_wave], yerr=sdv_2[good_wave], fmt='o', label='Side 2 BERV [-26,-24] km/s', alpha=0.2)
plt.xlabel('XPIX Wavelength Bin')
plt.ylabel('Average Radial Velocity (km/s)')
plt.title('Average Radial Velocity per Wavelength Bin for Different BERV Sides')
plt.legend()
plt.grid()
plt.show()


valid_1 = ~np.isnan(dv_1) & ~np.isnan(sdv_1) & (sdv_1>0)
valid_2 = ~np.isnan(dv_2) & ~np.isnan(sdv_2) & (sdv_2>0)

# fit a slope to dv_1 and dv_2 vs tbl_wave_start with errors sdv_1 and sdv_2
coeffs_1, cov_1 = np.polyfit(tbl_wave_start[valid_1], dv_1[valid_1], 1, w=1/sdv_1[valid_1], cov=True)
coeffs_2, cov_2 = np.polyfit(tbl_wave_start[valid_2], dv_2[valid_2], 1, w=1/sdv_2[valid_2], cov=True)

print('Slope Side 1:', coeffs_1[0], '±', np.sqrt(cov_1[0,0]))
print('Slope Side 2:', coeffs_2[0], '±', np.sqrt(cov_2[0,0]))

plt.errorbar(tbl_wave_start, dv_1, yerr=sdv_1, fmt='o', label='Side 1 BERV [-28,-26] km/s', alpha=0.2)
plt.errorbar(tbl_wave_start, dv_2, yerr=sdv_2, fmt='o', label='Side 2 BERV [-26,-24] km/s', alpha=0.2)
plt.xlabel('Wavelength Start (nm)')
plt.ylabel('Average Radial Velocity (km/s)')
plt.title('Average Radial Velocity per Wavelength Bin for Different BERV Sides')
plt.legend()
plt.grid()
plt.show()