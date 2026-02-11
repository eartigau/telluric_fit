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

import tellu_tools as tt
import warnings

project_path = tt.user_params()['project_path']

instrument = 'NIRPS'
obj = 'TOI4552'
batchname = 'skypca_v5'
# either self or model
template_style = 'model' #'self'

molecules = ['H2O', 'CH4', 'CO2', 'O2']

files = glob.glob(os.path.join(project_path,f'scidata_{instrument}/{obj}/NIRPS*.fits'))
hdr0 = fits.getheader(files[0])

# Reference/common wave grid used for alignment and plotting diagnostics
waveref = tt.getdata_safe(os.path.join(project_path,'calib_NIRPS/waveref.fits'))

if template_style == 'model':
    spl,spl_dv = tt.fetch_template(hdr0)
elif template_style == 'self':
    template_file = os.path.join(project_path,f'templates_{instrument}/Template_s1dv_{obj}_sc1d_v_file_A.fits')
    template = Table.read(template_file)
    wave_template = np.array(template['wavelength'])
    flux_template = np.array(template['flux'])
    flux_template/=mp.lowpassfilter(flux_template, 101)
    g = np.isfinite(flux_template)
    spl = ius(wave_template[g], flux_template[g], k=1, ext=0)
    grad = np.gradient(flux_template[g], wave_template[g])
    spl_dv = ius(wave_template[g], grad, k=1, ext=0)
else:
    raise ValueError(f'Unknown template style {template_style}')

batchname = batchname+'_'+template_style

sky_dict =tt.sky_pca_fast(sky_dict = None)


for file in files:
    print(f'\n\tProcessing {file} [{files.index(file)+1}/{len(files)}]')
    #file = files[len(files)//2]
    hdr0 = fits.getheader(file)

    # path to t.fits files
    file_id = hdr0['ARCFILE']

    t_name = os.path.join(project_path,f'orig_{instrument}',obj,file_id).replace('.fits','t.fits')
   
    # check if t_name exists
    if not os.path.exists(t_name):
        print(f'Skipping {file} as {t_name} does not exist.')
        continue

    # we fetch the extension 1 header and update it with the new exponents
    hdr = fits.getheader(t_name, ext=1)
    hdr = tt.update_header(hdr)

    # replace t.fits with patched_t.fits
    t_outname = t_name.replace('t.fits','tellupatched_t.fits').split('/')[-1]



    # check if corrected data path exists
    outpath = os.path.join(project_path,f'tellupatched_{instrument}/{obj}_{batchname}/')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    t_outname = os.path.join(outpath, t_outname)


    if os.path.exists(t_outname):
        print(f'Skipping {t_outname} as it already exists.')
        continue

    # make dirs if not exists
    t_outpath = os.path.dirname(t_outname)
    if not os.path.exists(t_outpath):
        os.makedirs(t_outpath)



    main_abso = fits.getdata(os.path.join(project_path,'main_absorber_NIRPS.fits'))

    abso_case = np.zeros_like(main_abso,dtype = int)
    abso_case[(main_abso ==0) | (main_abso ==4)] = 1 # we use the scaling with water vapour
    # if not (abso_case ==0) then we use the scaling without airmass
    abso_scaling = np.zeros_like(main_abso)

    hdr_tapas = fits.getheader(os.path.join(project_path,'LaSilla_tapas.fits'))

    model_file = 'params_fit_tellu_'+instrument+'.csv'
    model_table = Table.read(os.path.join(project_path, model_file))

    model = dict()
    for row in model_table:
        model[row['PARAM']] = row['VALUE']

    sp = fits.getdata(file)
    wavefile = hdr['WAVEFILE']
    wave = fits.getdata(os.path.join(project_path,f'calib_{instrument}/{wavefile}'))

    residual_intercept = np.zeros_like(sp)
    residual_slope = np.zeros_like(sp)
    residual_rms = np.zeros_like(sp)

    for iord in range(sp.shape[0]):
        intercept_file = os.path.join(project_path,f'residuals_{instrument}/residuals_order_{iord:02d}_intercept.fits')
        slope_file = os.path.join(project_path,f'residuals_{instrument}/residuals_order_{iord:02d}_slope.fits')
        rms_file = os.path.join(project_path,f'residuals_{instrument}/residuals_order_{iord:02d}_rms.fits')
        if glob.glob(intercept_file):
            residual_intercept[iord] = fits.getdata(intercept_file)
        if glob.glob(slope_file):
            residual_slope[iord] = fits.getdata(slope_file)
        if glob.glob(rms_file):
            residual_rms[iord] = fits.getdata(rms_file)

    hdr['PRESSURE'] = (hdr['HIERARCH ESO TEL AMBI PRES START']+hdr['HIERARCH ESO TEL AMBI PRES END'])/2.0
    hdr['AIRMASS'] = (hdr['HIERARCH ESO TEL AIRM START']+hdr['HIERARCH ESO TEL AIRM END'])/2.0
    hdr['TEMPERAT'] = hdr['HIERARCH ESO TEL AMBI TEMP']

    if False:
        iord = 60
        plt.plot(wave[iord], template2[iord]/np.nanpercentile(template2[iord],70), label='Template')
        plt.plot(wave[iord], sp[iord]/np.nanpercentile(sp[iord],70), label='Spectrum')
        plt.plot(wave[iord], (sp[iord]/template2[iord])/np.nanpercentile(sp[iord]/template2[iord],70), label='Ratio')
        plt.legend()
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized Flux')
        plt.title(f'Order {iord}')
        plt.show()

    airmass = hdr['AIRMASS']
    temperature = (hdr['TEMPERAT']+273.15)/273.15
    pressure_norm = hdr['PRESSURE']/hdr_tapas['PAMBIENT']
    mjd = hdr['MJDMID']

    params = [model['O2_AIRMASS_EXP'], model['O2_TEMPERATURE_EXP'],
            model['O2_PRESSURE_EXP'], model['O2_INTERCEPT']]

    o2_airm = airmass**params[0]*(temperature)**params[1]*(pressure_norm)**params[2]+params[3]
    expo_o2 = o2_airm*airmass*pressure_norm

    # now for the CO2
    params = [model['CO2_SLOPE'], model['CO2_INTERCEPT'], model['CO2_AMP'],
            model['CO2_PHASE'], model['CO2_AIRMASS_EXP'], model['CO2_TEMPERATURE_EXP'],
            model['CO2_PRESSURE_EXP']]

    co2_abso = (params[0]*mjd + params[1]  + params[2]*np.cos(2.0*np.pi*(mjd%365.24)/365.24 + params[3]))
    co2_abso = co2_abso*airmass**params[4]*(temperature)**params[5]*(pressure_norm)**params[6]
    expo_co2 = (co2_abso/hdr_tapas['VMR_CO2'])*airmass*pressure_norm

    # now for the CH4
    params = [model['CH4_SLOPE'], model['CH4_INTERCEPT'], model['CH4_AMP'],
            model['CH4_PHASE'], model['CH4_AIRMASS_EXP'], model['CH4_TEMPERATURE_EXP'],
            model['CH4_PRESSURE_EXP']]

    ch4_abso = (params[0]*mjd + params[1]  + params[2]*np.cos(2.0*np.pi*(mjd%365.24)/365.24 + params[3]))
    ch4_abso = ch4_abso*airmass**params[4]*(temperature)**params[5]*(pressure_norm)**params[6]
    expo_ch4 = (ch4_abso/hdr_tapas['VMR_CH4'])*airmass*pressure_norm

    msg = f'Initial exponents: CH4={expo_ch4:.3f}, CO2={expo_co2:.3f}, O2={expo_o2:.3f}'
    print(msg)

    sp0 = np.array(sp)

    sky_recon_final = np.zeros_like(sp)

    expos0 = np.array([None,expo_ch4, expo_co2, expo_o2])
    expos_no_water = np.array([0.0,expo_ch4, expo_co2, expo_o2])
    # TODO -- here we would divide by sp by the template shifted at the appropriate BERV velocity
    # TODO -- we may want to have a fine-tuning of the velocity with a rough estimate in case
    # TODO -- there is a large keplerian excursion.

    all_abso = tt.construct_abso(wave, expos0, all_abso=None)

    abso_no_water = tt.construct_abso(wave, expos_no_water, all_abso=all_abso)
    abso_no_water[abso_no_water<tt.user_params()['knee']] = np.nan
    sp_corr_tmp = (sp/abso_no_water)


    #sky_tmp = tt.recon_sky(wave=wave, spectrum=sp, sky_dict = sky_dict, force_positive=False)
    sp_no_sky = mp.lowpassfilter(sp_corr_tmp.ravel(),101).reshape(sp.shape)*abso_no_water
    sky_tmp = tt.sky_pca_fast(wave=wave, spectrum=sp-sp_no_sky, sky_dict = sky_dict, force_positive=True, doplot=tt.user_params()['doplot'])
    
    expos = tt.optimize_exponents(wave, sp-sky_tmp, airmass, fixed_exponents = expos0)

    abso = tt.construct_abso(wave, expos, all_abso=all_abso)
    abso[abso<tt.user_params()['knee']] = np.nan

    sp_tmp =  (sp-sky_tmp)/abso 

    veloshift = tt.get_velo(wave, sp_tmp, spl, dv_amp=200, doplot=tt.user_params()['doplot'])

    hdr['ABS_VELO'] = veloshift, 'BERV + systemic velocity (km/s)'
    hdr['SYS_VELO'] =  hdr['ABS_VELO']-hdr['BERV'],'Systemic velocity (km/s)'

    template2 = spl(wave*mp.relativistic_waveshift(veloshift))
    # only keep the domain to be fitted
    template2[wave < np.nanmin(tt.user_params()['wave_fit'])] = np.nan
    template2[wave > np.nanmax(tt.user_params()['wave_fit'])] = np.nan


    bad_orders = []
    for iord in range(sp.shape[0]):
        ratio = (sp_tmp[iord]/template2[iord])
        

        if np.mean(np.isfinite(ratio)) < 0.1:
            ratio = np.nan    
            bad_orders.append(iord)
        else:
            ratio[np.abs(ratio)>3*np.nanmedian(ratio)]=np.nan
            ratio[np.abs(ratio)<0.3*np.nanmedian(ratio)]=np.nan
            ratio = mp.lowpassfilter(ratio,501)
        template2[iord]*=ratio
    if len(bad_orders)>0:
        txt_bad = ', '.join([str(b) for b in bad_orders])
        print(f'  - Warning: bad orders detected: {txt_bad}')


    with warnings.catch_warnings(record=True) as _:
        low_flux_order =  np.nanmedian(template2,axis=1)<0.2*np.nanmedian(template2)
        template2[low_flux_order,:] = np.nan

    
    sp_recon = template2*abso

    sky_recon_final = tt.sky_pca_fast(wave=wave, spectrum=(sp-sp_recon), sky_dict = sky_dict, force_positive=True, doplot=tt.user_params()['doplot'])

    with warnings.catch_warnings(record=True) as _:
        expos = tt.optimize_exponents(wave, (sp-sky_recon_final)/template2, airmass, fixed_exponents = expos0)
    abso = tt.construct_abso(wave, expos, all_abso=all_abso)


    abso[abso<tt.user_params()['knee']] = np.nan

    sp_corr = (sp-sky_recon_final)/abso

    abso_scaling[abso_case==0] = hdr['AIRMASS']
    abso_scaling[abso_case==1] = expos[0] # the 0th is the water vapour exponent

    post_correction_waveref = residual_intercept + residual_slope*abso_scaling
    post_correction_waveref[~np.isfinite(post_correction_waveref)] = 0.0

    # send post_correction to the grid of sp and not the waveref
    post_correction = wavecore.wave_to_wave(post_correction_waveref, waveref, wave)
    post_correction[np.abs(post_correction) > np.exp(1)] = np.nan

    sp_corr/=np.exp(post_correction)

    # when above 1, set to nan, we lose > sqrt(2) in SNR
    reject_bright_sky = sky_recon_final / sp_corr > 1
    sp_corr[reject_bright_sky] = np.nan
    sp[reject_bright_sky] = np.nan

    # copy t_file to tellupatched location
    shutil.copyfile(t_name, t_outname)

    pressure = hdr['PRESSURE']
    pressure0 = hdr['NORMPRES']
    h2ocv = expos[0]*hdr['H2OCV']/(hdr['AIRMASS']*pressure/pressure0)
    co2_vmr = expos[1]*hdr['VMR_CO2']/(hdr['AIRMASS']*pressure/pressure0)
    ch4_vmr = expos[2]*hdr['VMR_CH4']/(hdr['AIRMASS']*pressure/pressure0)
    o2_frac = expos[3]/(hdr['AIRMASS']*pressure/pressure0)

    hdr['NORMPRES'] = pressure0,'[kPa] Normalization pressure for TAPAS values'
    hdr['H2O_CV'] = h2ocv,'[mm] at zenith, normalized pressure'
    hdr['CO2_VMR'] = co2_vmr, '[ppm] at zenith, normalized pressure'
    hdr['CH4_VMR'] = ch4_vmr,'[ppm] zenith, normalized pressure'
    hdr['O2_AIRM'] = o2_frac,'Airmass equivalent fraction at normalized pressure'

    # update header with optimized exponents
    for i, molecule in enumerate(molecules):
        hdr[f'EXPO_{molecule}'] = expos[i], f'Optimized exponent for {molecule}'

    #reject_O2 = tt.mask_o2(wave)
    #sp_corr[reject_O2] = np.nan
    #sp[reject_O2] = np.nan

    print(f'\n\tWe write to {t_outname} the telluric corrected spectrum.\n')

    # now update the telluric corrected spectrum into the correct extension of that file
    with fits.open(t_outname, mode='update') as hdul:
        if 'FluxA' in hdul:
            hdul['FluxA'].data = sp_corr
            hdul['FluxA'].header = hdr
        else:
            hdul.append(fits.ImageHDU(data=sp_corr, name='FluxA'))

        if 'Recon' in hdul:
            hdul['Recon'].data = (abso*np.exp(post_correction))
        else:
            hdul.append(fits.ImageHDU(data=(abso*np.exp(post_correction)), name='Recon'))

        hdul.flush()

    if tt.user_params()['doplot']:
        with warnings.catch_warnings(record=True) as _:
            abso_O2 = tt.construct_abso(wave, np.array([0.0,0.0,0.0,expo_o2]), all_abso=all_abso)

        fig, ax = plt.subplots(figsize=(12,6), nrows =2, ncols=1, sharex=True)
        #sp_old/=np.nanpercentile(sp_old,90)
        for iord in range(sp.shape[0]):
            with warnings.catch_warnings(record=True) as _:
                norm = np.nanpercentile(sp[iord],70)
            sp[iord]/=norm
            sp0[iord]/=norm
            sky_recon_final[iord]/=norm

            ax[1].plot(wave[iord], abso[iord], color='cyan', alpha=0.2, label=['Absorption',None][iord !=0])
            ax[1].plot(wave[iord], sp[iord], color='green', alpha=0.5, label=['Corrected Spectrum', None][iord !=0])
            ax[1].plot(wave[iord], sp0[iord], color='purple', alpha=0.5, label=['Original Spectrum', None][iord !=0])
            ax[1].plot(wave[iord], sp[iord]/abso[iord], color='red', alpha=0.9)
            ax[1].plot(wave[iord], sp[iord]/abso[iord]/np.exp(post_correction[iord]), color='black', alpha=0.9)
            ax[1].plot(wave[iord], post_correction[iord]/norm, color='orange', alpha=0.9)
            corr = (sp - sky_recon_final)[iord]/abso[iord]/np.exp(post_correction[iord])
            with warnings.catch_warnings(record=True) as _:
                corr/=template2[iord]
            corr/=np.nanpercentile(corr,70)

            ax[0].plot(wave[iord], corr, color='blue', alpha=0.5,label = ['APERO', None][iord !=0])
            ax[0].plot(wave[iord], sky_recon_final[iord], color='red', alpha=0.5,label = ['Sky Recon', None][iord !=0])
            ax[0].plot(wave[iord],abso_O2[iord], color='orange', alpha=0.2, label=['O2 Absorption', None][iord !=0])

        # load the drs data
        if False:
            scidata = fits.getdata('drs/r.NIRPS.2023-01-21T02:54:13.304_S2D_BLAZE_TELL_CORR_A.fits','SCIDATA')
            scidata[scidata <=0] = np.nan
            wavedata = fits.getdata('drs/r.NIRPS.2023-01-21T02:54:13.304_S2D_BLAZE_TELL_CORR_A.fits','WAVEDATA_VAC_BARY')/1e1
            wavedata = mp.relativistic_waveshift(wavedata, -hdr['BERV'])
            scidata/=np.gradient(wavedata,axis=1)
            scidata/=np.nanpercentile(scidata,70)

            for iord in range(scidata.shape[0]):
                ax[1].plot(wavedata[iord], scidata[iord]/np.nanpercentile(scidata[iord],70), color='blue', alpha=0.5)
                corr = scidata[iord]#/abso[iord]
                corr/=np.nanpercentile(corr,70)

                if iord ==0:
                    label = 'DRS'
                else:
                    label = None
                ax[0].plot(wavedata[iord], corr, color='green', alpha=0.5, label = label)

        if False:
            scidata = fits.getdata('apero_before/NIRPS_2023-01-21T02_54_13_304_pp_e2dsff_tcorr_A.fits')
            for iord in range(scidata.shape[0]):
                scidata[iord]/=np.nanpercentile(scidata[iord],70)
                ax[1].plot(wave[iord], scidata[iord], color='red', alpha=0.5)
                corr = scidata[iord]#/abso[iord]
                corr/=np.nanpercentile(corr,70)

                if iord ==0:
                    label = 'APERO before'
                else:
                    label = None
                ax[0].plot(wave[iord], corr, color='orange', alpha=0.5, label = label)

        ax[0].legend()
        ax[1].legend()
        ax[0].set_ylabel('Normalized flux')
        ax[0].set_ylim(0,2)
        ax[1].set_ylim(0,2)
        ax[1].set_ylabel('Normalized flux / Absorption')
        ax[1].set_xlabel('Wavelength (nm)')
        plt.show()
