#from pandas.core.arrays.arrow import dtype
from astropy.io import fits
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
# import curve fit
from scipy.optimize import curve_fit
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from aperocore import math as mp
import os
from aperocore.science import wavecore
from scipy.optimize import minimize
from scipy import constants
import shutil
from wpca import PCA, WPCA, EMPCA
from aperocore.science import wavecore

import tellu_tools as tt
import warnings
from tqdm import tqdm
from scipy.ndimage import binary_dilation

files = glob.glob('/Users/eartigau/test_fit/sky_NIRPS/NIRPS*A.fits')

waveref = fits.getdata('/Users/eartigau/test_fit/calib_NIRPS/waveref.fits')

sp0 = fits.getdata(files[0]).ravel()
map_sky = np.zeros((len(sp0),len(files)))
for ifile in tqdm(range(len(files)), desc='Processing files'):

    sp = fits.getdata(files[ifile])
    hdr = fits.getheader(files[ifile])
    wavefile =  hdr['WAVEFILE']
    wave = fits.getdata('/Users/eartigau/test_fit/calib_NIRPS/'+wavefile)

    sp = wavecore.wave_to_wave(sp, wave, waveref)

    for iord in tqdm(range(sp.shape[0]), desc='Processing orders', leave=False):
        for ite in range(3):
            n5,n95 = mp.nanpercentile(sp[iord],5), mp.nanpercentile(sp[iord],95)
            mask = np.zeros_like(sp[iord],dtype=float)
            mask[(sp[iord] < n5) | (sp[iord] > n95)] = np.nan
            sp[iord] -= mp.lowpassfilter(sp[iord]+mask,101)
        map_sky[iord*4088:(iord+1)*4088,ifile] = sp[iord]


p5,p95 = np.nanpercentile(map_sky,[5,95],axis=1)
moy = (p5+p95)/2.0

n1,s1 = np.nanpercentile(moy,[16,84])
sig = (s1-n1)/2.0

Npca = 10

map_sky[~np.isfinite(map_sky)] = 0.0
out = np.zeros((Npca, map_sky.shape[0]))

for YJ_H in range(2):
    # correct domain
    if YJ_H == 0:
        wavemin, wavemax = 950,1400
    else:
        wavemin, wavemax = 1400,1900

    # find which pixels are >3sig away from the mean
    mask_lines = (moy > 3.0*sig) & (waveref.ravel()>wavemin) & (waveref.ravel()<wavemax)
    # dilate by 3 pixels
    mask_lines = binary_dilation(mask_lines,iterations=3)


    pca = EMPCA(n_components=Npca).fit(map_sky[mask_lines,:].T)

    components = pca.components_.T

    out[:,mask_lines] = components.T

for icomponent in range(out.shape[0]):
    color = plt.cm.viridis(icomponent / (out.shape[0] - 1))
    plt.plot(waveref.ravel(),out[icomponent,:], color=color)
plt.show()

outname = '/Users/eartigau/test_fit/sky_NIRPS/sky_pca_components.fits'
fits.writeto(outname, out, overwrite=True)