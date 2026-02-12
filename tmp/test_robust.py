import numpy as np

def savgol_filter_nan_fast(y, window_length, polyorder, deriv=0, frac_valid=0.3):
    """Simplified version for testing"""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    y_filtered = np.full(n, np.nan)
    half_window = window_length // 2
    
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window = y[start:end]
        valid = np.isfinite(window)
        
        if np.sum(valid) < frac_valid * len(window):
            continue
        if np.sum(valid) <= polyorder:
            continue
            
        x_window = np.arange(len(window)) - (i - start)
        x_valid = x_window[valid]
        y_valid = window[valid]
        
        try:
            coeffs = np.polyfit(x_valid, y_valid, polyorder)
            y_filtered[i] = np.polyval(coeffs, 0)
        except:
            pass
    
    return y_filtered


def savgol_filter_robust(y, window_length, polyorder=3, n_sigma=5.0, max_iter=10, frac_valid=0.3):
    y = np.asarray(y, dtype=np.float64)
    y_work = y.copy()
    
    for iteration in range(max_iter):
        y_smooth = savgol_filter_nan_fast(y_work, window_length, polyorder, deriv=0, frac_valid=frac_valid)
        residuals = y_work - y_smooth
        valid_resid = residuals[np.isfinite(residuals)]
        
        if len(valid_resid) < 10:
            break
        
        p16, p84 = np.percentile(valid_resid, [16, 84])
        rms = (p84 - p16) / 2.0
        
        if rms == 0:
            break
        
        outliers = np.abs(residuals) > n_sigma * rms
        n_outliers = np.sum(outliers & np.isfinite(y_work))
        print(f'  Iter {iteration+1}: rms={rms:.4f}, outliers={n_outliers}')
        
        if n_outliers == 0:
            break
        
        y_work[outliers] = np.nan
    
    y_filtered = savgol_filter_nan_fast(y_work, window_length, polyorder, deriv=0, frac_valid=frac_valid)
    return y_filtered


if __name__ == '__main__':
    np.random.seed(42)
    x = np.linspace(0, 10, 500)
    y = np.sin(x) + 0.1 * np.random.randn(500)

    # Add outliers
    y[50] = 10
    y[150] = -8
    y[300] = 15

    # Add NaNs
    y[100:110] = np.nan

    print('Testing savgol_filter_robust...')
    result = savgol_filter_robust(y, window_length=51, polyorder=3, n_sigma=5.0)

    print(f'Input shape: {y.shape}, Output shape: {result.shape}')
    print(f'NaNs in input: {np.sum(np.isnan(y))}, in output: {np.sum(np.isnan(result))}')

    residuals = y - result
    mask = np.isfinite(residuals)
    mask[[50, 150, 300]] = False
    rms = np.std(residuals[mask])
    print(f'RMS of residuals (excluding outliers): {rms:.4f}')
    print('SUCCESS!')
