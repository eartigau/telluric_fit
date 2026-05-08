"""
Top-level driver for the telluric correction pipeline.

Runs the following steps in sequence for NIRPS or SPIROU:
  1. rsync       : sync data from remote server (sync_NIRPS or sync_SPIROU)
  2. slinky      : wavelength solution refinement (slinky_tools.run_slinky)
  3. hot-star    : fit hot-star spectra (smart_fit.py)
  4. compil_stats: compile stats, build main_absorber + params_fit_tellu (compil_stats.py)
  5. residuals   : build empirical per-pixel correction maps (residuals.py)
  6. telluric    : per-object telluric correction (predict_abso.main)

Usage
-----
    python run_pipeline.py                      # all steps; instrument read from telluric_config.yaml
    python run_pipeline.py --skip-sync          # skip rsync
    python run_pipeline.py --skip-slinky        # skip slinky
    python run_pipeline.py --skip-hotstar       # skip hot-star fit
    python run_pipeline.py --skip-compilstats   # skip compil_stats
    python run_pipeline.py --skip-residuals     # skip residuals
    python run_pipeline.py --only-slinky        # run only the slinky step
    python run_pipeline.py --only-telluric      # run only the telluric correction step
    python run_pipeline.py --object PROXIMA     # process a single object (default: science_targets from YAML)
    python run_pipeline.py --instrument SPIROU
"""

import argparse
import os
import subprocess
import sys
import time
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _tprint(msg, color=None):
    """Print a timestamped message."""
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def _load_telluric_config():
    cfg_path = os.path.join(SCRIPT_DIR, 'telluric_config.yaml')
    with open(cfg_path, 'r') as fh:
        return yaml.safe_load(fh)


def _step_header(title):
    bar = '=' * 60
    _tprint(bar)
    _tprint(f'  {title}')
    _tprint(bar)


# ---------------------------------------------------------------------------
# Step 1 — rsync
# ---------------------------------------------------------------------------

def run_sync(instrument):
    _step_header(f'STEP 1/6 — SYNC ({instrument})')
    sync_script = os.path.join(SCRIPT_DIR, f'sync_{instrument}')
    if not os.path.exists(sync_script):
        _tprint(f'Script {sync_script} not found — skipping sync.')
        return
    ret = subprocess.call(['bash', sync_script], cwd=SCRIPT_DIR)
    if ret != 0:
        _tprint(f'WARNING: sync_{instrument} returned exit code {ret}.')
    else:
        _tprint('Sync done.')


# ---------------------------------------------------------------------------
# Step 2 — slinky
# ---------------------------------------------------------------------------

def run_slinky():
    _step_header('STEP 2/6 — SLINKY (wavelength solution refinement)')
    import slinky_tools
    slinky_tools.run_slinky()
    _tprint('Slinky done.')


# ---------------------------------------------------------------------------
# Step 3 — hot-star fit
# ---------------------------------------------------------------------------

def run_hotstar(instrument):
    _step_header(f'STEP 3/6 — HOT-STAR FIT ({instrument})')
    import smart_fit as sf
    sf.main(instrument=instrument)
    _tprint('Hot-star fit done.')


# ---------------------------------------------------------------------------
# Step 4 — compil_stats
# ---------------------------------------------------------------------------

def run_compilstats(instrument):
    _step_header('STEP 4/6 — COMPIL_STATS (params_fit_tellu + main_absorber)')
    cs_script = os.path.join(SCRIPT_DIR, 'compil_stats.py')
    ret = subprocess.call([sys.executable, cs_script, instrument], cwd=SCRIPT_DIR)
    if ret != 0:
        _tprint(f'WARNING: compil_stats.py returned exit code {ret}.')
    else:
        _tprint('Compil_stats done.')


# ---------------------------------------------------------------------------
# Step 5 — residuals
# ---------------------------------------------------------------------------

def run_residuals():
    _step_header('STEP 5/6 — RESIDUALS (empirical per-pixel correction maps)')
    residuals_script = os.path.join(SCRIPT_DIR, 'residuals.py')
    ret = subprocess.call([sys.executable, residuals_script], cwd=SCRIPT_DIR)
    if ret != 0:
        _tprint(f'WARNING: residuals.py returned exit code {ret}.')
    else:
        _tprint('Residuals done.')


# ---------------------------------------------------------------------------
# Step 6 — predict_abso par objet
# ---------------------------------------------------------------------------

def run_telluric(objects, instrument, batch_name, template_style, force_recompute):
    _step_header(f'STEP 6/6 — TELLURIC CORRECTION ({instrument})')
    _tprint(f'Objects: {objects}')

    # Import here to avoid slow loading when only --help is requested
    import predict_abso as pa

    first = True
    for obj in objects:
        _tprint(f'--- Processing {obj} ---')
        pa.main(
            batch_name=batch_name,
            instrument=instrument,
            obj=obj,
            template_style=template_style,
            force_recompute=(force_recompute and first),
        )
        first = False

    _tprint('Telluric correction done.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = _load_telluric_config()
    default_instrument = cfg.get('instrument', 'NIRPS').upper()
    default_targets = sorted({
        t
        for info in cfg.get('data_recipients', {}).values()
        for t in (info if isinstance(info, list) else info.get('targets', []))
    })
    default_template = cfg.get('template_style', 'model')

    # Read batch name from telluric_config.yaml
    batch_section = cfg.get('batch', {})
    default_batch = (batch_section.get('name') if isinstance(batch_section, dict)
                     else cfg.get('batch_name', 'skypca_v5'))

    parser = argparse.ArgumentParser(
        description='Full telluric correction pipeline (rsync → slinky → residuals → telluric)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--instrument', default=default_instrument,
                        choices=['NIRPS', 'SPIROU'],
                        help='Instrument')
    parser.add_argument('--object', default=None,
                        help='Process a single object (default: all science_targets from YAML)')
    parser.add_argument('--batch', default=default_batch,
                        help='Batch name (output identifier)')
    parser.add_argument('--template', default=default_template,
                        choices=['model', 'self', 'smart'],
                        help='Stellar template style')

    # Step control
    step_group = parser.add_argument_group('Step control')
    step_group.add_argument('--skip-sync',        action='store_true', help='Skip the rsync step')
    step_group.add_argument('--skip-slinky',      action='store_true', help='Skip the slinky step')
    step_group.add_argument('--skip-hotstar',     action='store_true', help='Skip the hot-star fit step')
    step_group.add_argument('--skip-compilstats', action='store_true', help='Skip the compil_stats step')
    step_group.add_argument('--skip-residuals',   action='store_true', help='Skip the residuals step')
    step_group.add_argument('--skip-telluric',    action='store_true', help='Skip the telluric correction step')
    step_group.add_argument('--only-slinky',      action='store_true',
                            help='Run only the slinky step')
    step_group.add_argument('--only-telluric',    action='store_true',
                            help='Run only the telluric correction step (implies all other skips)')

    parser.add_argument('--recompute', action='store_true',
                        help='Force recomputation of the pre-computed absorption grid')

    args = parser.parse_args()

    if args.only_slinky:
        args.skip_sync = True
        args.skip_hotstar = True
        args.skip_compilstats = True
        args.skip_residuals = True
        args.skip_telluric = True

    if args.only_telluric:
        args.skip_sync = True
        args.skip_slinky = True
        args.skip_hotstar = True
        args.skip_compilstats = True
        args.skip_residuals = True

    instrument = args.instrument.upper()
    objects = [args.object] if args.object else default_targets

    if not objects:
        print('ERROR: no object specified and data_recipients is empty in telluric_config.yaml.')
        sys.exit(1)

    t0 = time.time()
    _tprint(f'Pipeline started | instrument={instrument} | objects={objects} | batch={args.batch}')

    # ---- Step 1 ----
    if not args.skip_sync:
        run_sync(instrument)

    # ---- Step 2 ----
    if not args.skip_slinky:
        run_slinky()

    # ---- Step 3 ----
    if not args.skip_hotstar:
        run_hotstar(instrument)

    # ---- Step 4 ----
    if not args.skip_compilstats:
        run_compilstats(instrument)

    # ---- Step 5 ----
    if not args.skip_residuals:
        run_residuals()

    # ---- Step 6 ----
    if not args.skip_telluric:
        run_telluric(
            objects=objects,
            instrument=instrument,
            batch_name=args.batch,
            template_style=args.template,
            force_recompute=args.recompute,
        )

    elapsed = time.time() - t0
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    _step_header(f'PIPELINE DONE — total elapsed time {h:02d}:{m:02d}:{s:02d}')


if __name__ == '__main__':
    main()
