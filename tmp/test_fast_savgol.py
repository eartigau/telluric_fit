import numpy as np
import time
from scipy.signal import savgol_filter

def _savgol_interp_nan(y, window_length, polyorder):
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(y)
    if not np.any(valid):
        return np.full_like(y, np.nan)
    if np.all(valid):
        return savgol_filter(y, window_length, polyorder, mode='nearest')
    x = np.arange(len(y))
    y_interp = np.interp(x, x[valid], y[valid])
    return savgol_filter(y_interp, window_length, polyorder, mode='nearest')

def savgol_filter_robust(y, window_length, polyorder=3, n_sigma=5.0, max_iter=10):
    y = np.asarray(y, dtype=np.float64)
    y_work = y.copy()
    for iteration in range(max_iter):
        y_smooth = _savgol_interp_nan(y_work, window_length, polyorder)
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
        if n_outliers == 0:
            break
        y_work[outliers] = np.nan
    return _savgol_interp_nan(y_work, window_length, polyorder)

if __name__ == '__main__':
    # Test with realistic size (4088 pixels like NIRPS)
    np.random.seed(42)
    y = np.sin(np.linspace(0, 20, 4088)) + 0.1 * np.random.randn(4088)
    y[500] = 10
    y[2000] = -8
    y[3500] = 15
    y[1000:1020] = np.nan

    t0 = time.time()
    for _ in range(100):
        result = savgol_filter_robust(y, 101, 3, 5.0)
    elapsed = time.time() - t0
    print(f'100 calls on 4088 pixels: {elapsed:.2f}s ({elapsed*10:.1f}ms per call)')
    print('SUCCESS!')
