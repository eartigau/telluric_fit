# Telluric Correction Pipeline Workflow

This document describes the complete workflow for processing spectroscopic data through the telluric correction pipeline. The pipeline consists of multiple sequential steps that progressively refine the telluric correction models.

## Overview

The pipeline has five main stages:

1. **Data Synchronization**: Copy calibration and science data from remote servers
2. **Hot Star Telluric Fitting**: Determine telluric transmission from hot star spectra
3. **Statistics Compilation**: Aggregate telluric parameters across all observations
4. **Residual Modeling**: Build per-pixel empirical correction models from residuals
5. **Per-Object Correction**: Apply telluric correction to science targets

---

## Step 1: Data Synchronization

### Command
```bash
bash sync_NIRPS   # For NIRPS data
bash sync_SPIROU  # For SPIRou data
```

### What It Does
The `sync_{INSTRUMENT}` script uses `rsync` to copy required data files from remote servers to the local project directory:

- **Calibration data**: Reference files (waveref, blaze, etc.) from `calib/` remote directory
  - Destination: `calib_NIRPS/` or `calib_SPIROU/`
  - Files: `*_e2dsff_A_wave_night_A.fits`

- **Science data**: Raw spectra for all science targets
  - Destination: `scidata_NIRPS/` or `scidata_SPIROU/`
  - Files: `**/*_e2dsff_A.fits` (recursive, preserves directory structure)

- **Hot star data**: Reference hot stars for telluric fitting (flattened structure)
  - Destination: `hotstars_NIRPS/` or `hotstars_SPIROU/`
  - Objects: Defined in `telluric_config.yaml` under `hot_stars:` key
  - Current list includes: `15PEG`, `17PEG`, `23LMI`, `24LYN`, `31CAS`, `33LYN`, `51DRA`, `59PEG`, `74PSCB`, `82UMA`, `BETSER`, `CHICAP`, `ETAPYX`, `GAMSCT`, `GAMTRI`, `HD130917`, `HD159170`, `HD195094`, `HR1314`, `HR1832`, `HR1903`, `HR2180`, `HR2209`, `HR3117`, `HR3131`, `HR3314`, `HR4023`, `HR4467`, `HR4468`, `HR4687`, `HR4722`, `HR4889`, `HR5107`, `HR5671`, `HR6025`, `HR6743`, `HR7590`, `HR7830`, `HR806`, `HR8489`, `HR8709`, `HR9098`, `IOTCYG`, `LLEO`, `OMICAPA`, `PHILEO`, `PI02ORI`, `ZETLEP`, `ZETVIR`
  - Files: All `*_e2dsff_A.fits` files matching these objects

### Output
- Populated directories with all necessary calibration and science data
- Ready for telluric fitting

---

## Step 2: Hot Star Telluric Fitting

### Command
```bash
python smart_fit.py
```

### What It Does

This script processes all hot star spectra to determine the telluric transmission function. Hot stars have simple, featureless spectra in the infrared, making them ideal for measuring atmospheric absorption.

#### Key Operations:

1. **Load calibration data**
   - Reference wavelength grid
   - Blaze function (instrument response)
   - Sky PCA components (for OH airglow removal)

2. **For each hot star spectrum**:
   - Resample to reference wavelength grid
   - Normalize to continuum (90th percentile)
   - Remove sky emission using PCA decomposition
   - Fit absorption exponents for atmospheric molecules

3. **Atmospheric molecules fitted**:
   - H₂O (water vapor)
   - CH₄ (methane)
   - CO₂ (carbon dioxide)
   - O₂ (oxygen, A-band)

4. **Exponent optimization**:
   - Uses airmass-dependent absorption model: τ = τ₀ × airmass^exponent
   - Optimizes exponents by minimizing spectral gradient
   - Each molecule's exponent fitted independently

5. **Header updates**:
   - Column densities: `H2OCV`, `VMR_CO2`, `VMR_CH4`
   - Optimized exponents
   - Telluric correction metadata

### Output Files
- `tellu_fit_NIRPS/trans_*.fits`: Telluric transmission spectra
- Updated FITS headers with telluric parameters

### Example Output
```
11:39:30.272 | ******************** HD195094 [1/10] ********************
11:39:30.273 | Processing: tellu_fit_NIRPS/trans_NIRPS_2024-12-28T00_56_24_207_pp_e2dsff_A.fits
11:39:39.798 | Starting exponent optimization for ['H2O', 'CH4', 'CO2', 'O2']...
11:39:40.389 | Exponent optimization complete
Optimized expo for H2O: 0.6560
Optimized expo for CH4: 1.0150
Optimized expo for CO2: 1.0088
Optimized expo for O2: 0.9967
Airmass used: 1.034
```

---

## Step 3: Statistics Compilation

### Command
```bash
python compil_stats.py
```

### What It Does

This script aggregates telluric parameters from all hot star observations to create statistical summaries and quality metrics.

#### Key Operations:

1. **Read telluric-corrected hot star headers**
   - Extract exponent optimizations
   - Extract column density measurements
   - Extract observation conditions (airmass, seeing, etc.)

2. **Compute statistics**:
   - Mean and standard deviation of exponents
   - Trend analysis (time-dependent variations)
   - Correlation with observing conditions
   - Outlier detection

3. **Quality assessment**:
   - Identify unreliable measurements
   - Flag observations with poor convergence
   - Quantify atmospheric stability

4. **Generate reports**:
   - Statistical summaries
   - Diagnostic plots
   - Parameter trends over time

### Output Files
- `params_fit_tellu_NIRPS.csv`: Fitted atmospheric model parameters
- Plots of parameter distributions (O2, CO2, CH4, H2O)
- Quality flags for subsequent processing

---

## Step 4: Residual Modeling

### Command
```bash
python residuals.py
```

### What It Does

This script builds per-pixel empirical correction models to account for systematic residuals in the telluric model that are not captured by the TAPAS-based absorption templates.

#### Key Operations:

1. **Load all fitted transmissions**
   - Read `trans_*.fits` files from tellu_fit
   - Build a large cube of all residual spectra

2. **Per-object alignment**
   - Align residuals to common wavelength grid using BERV
   - Remove median residual pattern per object (stellar features)
   - Detrend each exposure by wavelength

3. **Per-pixel linear modeling**
   - For H2O/no-absorption pixels: fit slope vs EXPO_H2O
   - For O2/CO2/CH4 pixels: fit slope vs AIRMASS (morning only)
   - Compute RMS of residuals after model subtraction

4. **Quality filtering**
   - Use only hot stars with EXPO_H2O < 7.0
   - Require >50% valid pixels for fitting

### Output Files
- `residuals_NIRPS/residuals_order_XX_slope.fits`: Slope correction per pixel
- `residuals_NIRPS/residuals_order_XX_intercept.fits`: Intercept correction per pixel  
- `residuals_NIRPS/residuals_order_XX_rms.fits`: RMS after correction per pixel

### Why This Step Matters

Without this step, `predict_abso.py` will still run but the post-correction will not be applied. The residual modeling captures:
- Wavelength-dependent errors in TAPAS line positions/depths
- Systematic biases in the blaze correction
- Instrumental effects not modeled by TAPAS

---

## Step 5: Per-Object Telluric Correction

### Command
```bash
python predict_abso.py --config objects_config.yaml
```

### What It Does

This script applies the telluric correction to science target spectra using the models determined from hot stars. Processing multiple objects is controlled via a YAML configuration file.

#### Configuration File Format (`objects_config.yaml`)

```yaml
# Telluric correction batch processing configuration
batch_name: "batch_v1"
instrument: "NIRPS"
template_style: "model"  # or "self" for self-correlation

# List of objects to process
objects:
  - name: "TOI4552"
    template_style: "model"
  - name: "WASP103"
    template_style: "model"
  - name: "HD1234"
    template_style: "self"

# Processing parameters
molecules: ["H2O", "CH4", "CO2", "O2"]
lowpass_filter_size: 101
template_ratio_threshold_high: 3.0
template_ratio_threshold_low: 0.3
template_smooth_window: 501
sky_rejection_threshold: 1.0
dv_amp: 200  # km/s velocity search range

# Advanced options
verbose: true
doplot: false
```

#### Key Operations:

1. **Load models from hot star fitting**
   - Atmospheric exponents
   - Column densities
   - Transmission functions

2. **For each science target**:
   - Load spectrum and reference wavelength
   - Load or generate template (stellar model or self)
   - Estimate radial velocity
   - Compute telluric transmission at observation conditions
   - Remove telluric absorption from spectrum
   - Remove sky emission

3. **Output corrected spectra**:
   - Telluric-corrected science spectra
   - Updated FITS headers with correction metadata
   - Diagnostic information

### Output Files
- `tellu_corrected_NIRPS/`: Telluric-corrected science spectra
- Updated FITS headers with:
  - Applied correction parameters
  - Residual telluric features
  - Signal-to-noise ratios

---

## Processing Pipeline Architecture

### Data Flow Diagram

```
Remote Data
    ↓
[sync] → calib_NIRPS/, scidata_NIRPS/, hotstars_NIRPS/
    ↓
[smart_fit.py] → tellu_fit_NIRPS/ (hot star transmission)
    ↓
[compil_stats.py] → params_fit_tellu_NIRPS.csv (atmospheric models)
    ↓
[residuals.py] → residuals_NIRPS/ (per-pixel correction maps)
    ↓
objects_config.yaml (batch configuration)
    ↓
[predict_abso.py] → tellu_corrected_NIRPS/ (science data)
```

---

## Important Notes

### Color-Coded Output
The pipeline provides color-coded console output for easy monitoring:
- **Green**: Progress messages about what the pipeline is doing
- **Blue**: Numerical values and parameters
- **Cyan**: Hot-star banners with progress [n/N]
- **Orange**: Warnings or issues requiring attention

### Error Handling
If a step fails:
1. Check the error message (timestamps help locate issues)
2. Verify input data availability and integrity
3. Review step-specific logs
4. Correct the issue and re-run that step

### Checkpoints
- Each step produces output files that serve as checkpoints
- You can restart from any step by ensuring prior outputs exist
- Do not delete intermediate files until the pipeline completes successfully

---

## Quick Reference

| Step | Command | Input | Output |
|------|---------|-------|--------|
| 1. Sync | `bash sync_{INSTRUMENT}` | Remote servers | `calib_*/`, `scidata_*/`, `hotstars_*/` |
| 2. Hot Star Fit | `python smart_fit.py` | Hot star spectra | `tellu_fit_NIRPS/` |
| 3. Statistics | `python compil_stats.py` | Hot star headers | `params_fit_tellu_NIRPS.csv` |
| 4. Residuals | `python residuals.py` | `tellu_fit_NIRPS/` | `residuals_NIRPS/` |
| 5. Science Correction | `python predict_abso.py --config config.yaml` | Science spectra + config | `tellu_corrected_NIRPS/` |

---

## Troubleshooting

### Common Issues

**Missing calibration files**
- Re-run the sync step
- Check network connectivity to remote servers

**Exponent optimization fails**
- Verify hot star spectrum quality
- Check that sky subtraction worked properly
- Review airmass values

**Science correction produces artifacts**
- Verify template quality
- Check radial velocity determination
- Review telluric model validity

For detailed debugging, enable verbose output in configuration files.
---

## Telluric Masking Configuration

The pipeline uses per-molecule depth thresholds to mask unreliable pixels in telluric-corrected spectra. This is configured in `telluric_config.yaml`.

### Configuration File (`telluric_config.yaml`)

```yaml
# Weight transition parameters
weighting:
  transition_sigma: 0.02  # Smaller = sharper transition

# Per-molecule parameters
molecules:
  H2O:
    depth_max: 0.5        # Reject pixels below this transmission
    depth_saturated: 0.2  # Pixels below this contaminate neighbors
    reject_saturated: 0.8 # Contamination extends until this threshold
  CH4:
    depth_max: 0.5
    depth_saturated: 0.2
    reject_saturated: 0.8
  CO2:
    depth_max: 0.5
    depth_saturated: 0.2
    reject_saturated: 0.8
  O2:
    depth_max: 0.8        # O2 lines are well-modeled, allow deeper
    depth_saturated: 0.2
    reject_saturated: 0.9
```

### How It Works

1. **`depth_max`**: Pixels with transmission below this threshold are masked. A 50% absorption (trans=0.5) may have significant modeling errors.

2. **`depth_saturated`**: Pixels below this threshold are considered "saturated" and contaminate neighboring pixels. Even if a neighbor has 70% transmission, it's unreliable if it's next to a 10% pixel.

3. **`reject_saturated`**: The contamination from saturated pixels extends outward until transmission reaches this threshold. This masks the wings of deep absorption lines.

### Example

```
Transmission: [0.9, 0.75, 0.1, 0.6, 0.9]
                           ↑ saturated (< 0.2)

With depth_max=0.5, depth_saturated=0.2, reject_saturated=0.8:
  - Pixel 3 (0.1): masked (< depth_saturated)
  - Pixel 2 (0.75): masked (contaminated, < reject_saturated)
  - Pixel 4 (0.6): masked (contaminated, < reject_saturated)
  - Pixels 1,5 (0.9): valid (>= reject_saturated, stops contamination)

Result: [True, False, False, False, True]
```

### Usage in Code

```python
import tellu_tools as tt

# During optimization (smooth weights, no hard cutoff)
trans = tt.construct_abso(wave, expos, all_abso, apply_final_mask=False)
weights = tt.construct_abso.last_weights  # Smooth 0-1 weights

# For final corrected spectrum (hard mask applied)
trans_final = tt.construct_abso(wave, expos, all_abso, apply_final_mask=True)
corrected = spectrum / trans_final  # NaN where unreliable

# Get weights for a specific molecule
h2o_weights = tt.get_molecule_weights(h2o_transmission, 'H2O')

# Or use the low-level function with custom parameters
weights = tt.get_transmission_weights(
    transmission,
    depth_max=0.5,
    depth_saturated=0.2,
    reject_saturated=0.8,
    transition_sigma=0.02
)
```

### Smooth vs Hard Masking

- **During optimization**: Smooth weights (0→1) prevent convergence oscillations. The optimizer sees gradual changes rather than discrete jumps.

- **For final output**: Pixels with weight < 0.5 are set to NaN. The sharp transition (`transition_sigma=0.02`) makes this nearly binary while remaining mathematically smooth.

---

## Machine-Specific Configuration

The pipeline supports running on different machines with different data paths. This is configured in `batch_config.yaml`.

### Configuration File (`batch_config.yaml`)

```yaml
machines:
  FIR:
    detect_path: /cosmos99/nirps/apero-data/common/
    project_path: /cosmos99/nirps/apero-data/common/telluric_night/
    doplot: false
  MacBook-eartigau:
    detect_path: /Users/eartigau/nirps/
    project_path: /Users/eartigau/nirps/
    doplot: true

batch:
  name: skypca_v5
  # ... other batch settings
```

### How It Works

On startup, the pipeline checks which `detect_path` exists and uses the corresponding `project_path`. This allows the same code to run on:

- **FIR server**: Production processing with large data volumes (`doplot: false`)
- **Local MacBook**: Development and testing with interactive plots (`doplot: true`)

### Detection Logic

```python
# In tellu_tools_config.py
for machine_name, machine_config in config['machines'].items():
    if os.path.exists(machine_config['detect_path']):
        return machine_config['project_path']
```

---

## Precomputed Absorption Grid

The most expensive operation in the pipeline is convolving absorption spectra with the instrumental resolution. To speed this up, the pipeline precomputes convolved absorption at discrete exponent values on first run.

### How It Works

1. **First run**: Computes ~500 templates (convolved absorption at exponent steps of 0.05)
   - H₂O: exponents 0.2 to 20.0 (397 templates)
   - CH₄, CO₂, O₂: exponents 0.9 to 3.0 (43 templates each)
   - Takes ~2-3 minutes, saves to pickle file

2. **Subsequent runs**: Loads from pickle (<1 second)
   - Interpolates linearly between bracketing exponents
   - Single spline to observation wavelength grid

### Storage Location

```
{project_path}/tmp_{INSTRUMENT}/precomputed_absorption_grid.pkl
```

Each machine stores its own pickle (different project_path).

### Automatic Recomputation

The pickle stores the molecule parameters from `telluric_config.yaml`. If you change depth_max, depth_saturated, or reject_saturated, the grid is automatically recomputed on next run.

### Manual Recomputation

```python
import tellu_tools as tt
tt.precompute_absorption_grid(instrument='NIRPS', force_recompute=True)
```

### Performance Gain

| Operation | Without Precomputation | With Precomputation |
|-----------|----------------------|-------------------|
| Per-file optimization | ~10s | ~0.5s |
| 100 files | ~17 min | ~1 min |

---

## Python API Usage

In addition to command-line usage, you can call `predict_abso.py` programmatically:

### Basic Usage

```python
from predict_abso import main

main(
    batch_name='skypca_v5',
    instrument='NIRPS',
    obj='TOI4552',
    template_style='model'
)
```

### Processing Multiple Objects

```python
from predict_abso import main

objects = ['TOI4552', 'TOI1234', 'HD189733']

for obj in objects:
    print(f"\n{'='*60}\nProcessing {obj}\n{'='*60}")
    
    try:
        main(
            batch_name='batch_multi',
            instrument='NIRPS',
            obj=obj,
            template_style='model'
        )
    except Exception as e:
        print(f"Error for {obj}: {e}")
        continue
```

### Configurable Parameters

Parameters can be modified in `predict_abso_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lowpass_filter_size` | 101 | Size of lowpass filter window |
| `template_ratio_threshold_high` | 3.0 | High threshold for outlier rejection |
| `template_ratio_threshold_low` | 0.3 | Low threshold for outlier rejection |
| `template_smooth_window` | 501 | Template ratio smoothing window |
| `min_valid_ratio` | 0.1 | Minimum fraction of valid pixels |
| `low_flux_threshold` | 0.2 | Low flux rejection threshold |
| `sky_rejection_threshold` | 1.0 | Sky rejection threshold |
| `dv_amp` | 200 | Velocity search amplitude (km/s) |

---

## Verifying Results

### Check Output Files

```python
import glob
import os
from astropy.io import fits

output_dir = 'tellupatched_NIRPS/TOI4552_skypca_v5_model/'
files = sorted(glob.glob(os.path.join(output_dir, '*tellupatched_t.fits')))

print(f"Files processed: {len(files)}")

if files:
    with fits.open(files[0]) as hdul:
        print("\nFITS extensions:")
        hdul.info()
        
        print("\nAdded keywords:")
        hdr = hdul['FluxA'].header
        for key in ['ABS_VELO', 'SYS_VELO', 'EXPO_H2O', 'EXPO_CO2',
                    'EXPO_CH4', 'EXPO_O2', 'H2O_CV', 'CO2_VMR']:
            if key in hdr:
                print(f"  {key}: {hdr[key]}")
```

### Quick Visualization

```python
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

file = 'tellupatched_NIRPS/TOI4552_test/file_tellupatched_t.fits'

with fits.open(file) as hdul:
    sp_corr = hdul['FluxA'].data
    recon = hdul['Recon'].data
    wave = fits.getdata('calib_NIRPS/waveref.fits')

# Plot a single order
iord = 40

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax[0].plot(wave[iord], sp_corr[iord], 'k-', alpha=0.7, label='Corrected')
ax[0].set_ylabel('Flux')
ax[0].legend()

ax[1].plot(wave[iord], recon[iord], 'r-', alpha=0.7, label='Absorption')
ax[1].set_ylabel('Transmission')
ax[1].set_xlabel('Wavelength (nm)')
ax[1].legend()

plt.tight_layout()
plt.savefig('check_correction.png', dpi=150)
```

---

## Performance Benchmarks

Typical processing times on MacBook Pro M1:

| Number of files | Total time | Time/file |
|-----------------|------------|-----------|
| 10 | ~15 min | ~1.5 min |
| 50 | ~75 min | ~1.5 min |
| 100 | ~150 min | ~1.5 min |

**Factors affecting performance:**
- Number of spectral orders
- Number of optimization iterations
- Plot generation (`doplot=True` slows down processing)
- Disk I/O speed

---

## Best Practices

### Batch Naming Convention

Use descriptive, versioned batch names:

```
batch_name = "{purpose}_{version}"

Examples:
- "skypca_v5"
- "test_new_algo_v1"
- "paper_final_v3"
```

### Traceability

Always save the configuration used for processing:

```python
import json
from datetime import datetime

config = get_batch_config(...)
config['processing_date'] = datetime.now().isoformat()
config['user'] = os.environ.get('USER', 'unknown')

with open(f"config_{config['batch_name']}.json", 'w') as f:
    json.dump(config, f, indent=2)
```

### Validation Checklist

Before processing large datasets:

1. Test on 1-2 files first
2. Visually inspect results
3. Compare with previous version if available
4. Validate optimized exponents are reasonable
5. Document the configuration used

### Data Backup

Always keep original data and backup processed results:

```bash
# Backup processed data
tar -czf tellupatched_backup_$(date +%Y%m%d).tar.gz tellupatched_NIRPS/
```

---

## Git Repository

Clone the repository:

```bash
git clone https://github.com/eartigau/telluric_fit.git
cd telluric_fit
git pull  # to get latest updates
```