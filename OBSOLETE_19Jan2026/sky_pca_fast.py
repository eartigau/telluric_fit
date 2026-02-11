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
            sky = np.dot(amps, cube_dom)
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
                sky_unclipped = np.dot(amps, cube_dom)
                weighted_res = weighted_res * (sky_unclipped >= 0)
            
            grad = -2.0 * np.dot(cube_dom, weighted_res) / n_valid
            
            return obj, grad
        
        # L-BFGS-B optimization with analytical gradient
        result = minimize(
            objective_and_gradient,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if verbose:
            print(f'\t  Converged: {result.success}, iterations: {result.nit}, '
                  f'Chi2: {result.fun:.4f}')
        
        # Apply final model
        sky_out[domain] += compute_sky(result.x)
    
    if doplot:
        plt.figure(figsize=(12, 4))
        plt.plot(wave_flat, spectrum_flat, 'b', alpha=0.5, lw=0.5, label='Spectrum')
        plt.plot(wave_flat, sky_out, 'r', alpha=0.8, lw=0.5, label='Sky model')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return sky_out.reshape(spectrum.shape)
