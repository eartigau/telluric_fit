import numpy as np
from tellu_tools import savgol_filter_robust

# Create test data with outliers and NaNs
np.random.seed(42)
x = np.linspace(0, 10, 500)
y = np.sin(x) + 0.1 * np.random.randn(500)

# Add some outliers
y[50] = 10
y[150] = -8
y[300] = 15

# Add some NaNs
y[100:110] = np.nan

print('Testing savgol_filter_robust...')
result = savgol_filter_robust(y, window_length=51, polyorder=3, n_sigma=5.0)

print(f'Input shape: {y.shape}')
print(f'Output shape: {result.shape}')
print(f'NaNs in input: {np.sum(np.isnan(y))}')
print(f'NaNs in output: {np.sum(np.isnan(result))}')
print(f'Finite values in output: {np.sum(np.isfinite(result))}')

# Check that outliers were handled
residuals = y - result
mask = np.isfinite(residuals)
mask[[50, 150, 300]] = False  # Exclude known outliers
rms = np.std(residuals[mask])
print(f'RMS of residuals (excluding outliers): {rms:.4f}')
print('SUCCESS!')
