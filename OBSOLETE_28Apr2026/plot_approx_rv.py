from astropy.io import fits
import numpy as np
import glob
from astropy.table import Table
import matplotlib.pyplot as plt


path1 = '/Users/eartigau/test_fit/tellupatched_NIRPS/TOI756_skypca_v5_model/'
path2 = '/Users/eartigau/test_fit/tellupatched_NIRPS/TOI756_cutsky1/'

files = glob.glob(path1 + '*.fits') + glob.glob(path2 + '*.fits')
files = np.sort(files)

tbl = Table()
tbl['FILE'] = files
tbl['rjd'] = 0.0
tbl['vrad'] = 0.0
tbl['EXPO_H2O'] = 0.0
tbl['BERV'] = 0.0
tbl['PATHID'] = np.zeros(len(files), dtype=int)

for i in range(len(files)):
    hdr = fits.getheader(files[i], ext=1)
    tbl['rjd'][i] = hdr['MJDMID']
    tbl['vrad'][i] = hdr['SYS_VELO']*1000
    tbl['EXPO_H2O'][i] = hdr['EXPO_H2O']
    tbl['BERV'][i] = hdr['BERV']*1000

    for ipath in range(2):
        if path1 in files[i] and ipath == 0:
            tbl['PATHID'][i] = 0
        if path2 in files[i] and ipath == 1:
            tbl['PATHID'][i] = 1

tbl['svrad'] = 20.0


tbl0 = Table()
tbl0['rjd'] = tbl['rjd']
tbl0['vrad'] = tbl['vrad']
tbl0['svrad'] = tbl['svrad']
ord = np.argsort(tbl0['rjd'])
tbl0 = tbl0[ord]
tbl0.write('approx_rv_table.rdb', overwrite=True, format='rdb')


print(np.nanstd(tbl['vrad']))

k1 = tbl['PATHID'] == 0
k2 = tbl['PATHID'] == 1
plt.plot(tbl['BERV'][k1], tbl['vrad'][k1], 'o', label='Path 1',alpha=0.5)
plt.plot(tbl['BERV'][k2], tbl['vrad'][k2], 'o', label='Path 2',alpha=0.5)
plt.xlabel('BERV')
plt.ylabel('Radial Velocity (km/s)')
plt.title('Radial Velocity vs BERV')
plt.legend()
plt.grid()
plt.show()

k1 = tbl['PATHID'] == 0
k2 = tbl['PATHID'] == 1
plt.plot(tbl['rjd'][k1], tbl['vrad'][k1], 'o', label='Path 1',alpha=0.5)
plt.plot(tbl['rjd'][k2], tbl['vrad'][k2], 'o', label='Path 2',alpha=0.5)
plt.xlabel('RJD')
plt.ylabel('Radial Velocity (km/s)')
plt.title('Radial Velocity vs RJD')
plt.legend()
plt.grid()
plt.show()

plt.plot(tbl['rjd'], tbl['vrad'], 'o')
plt.xlabel('RJD')
plt.ylabel('Radial Velocity (km/s)')
plt.title('Radial Velocity vs RJD')
plt.grid()
plt.show()