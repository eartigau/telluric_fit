"""
Superscript du pipeline de correction tellurique.

Enchaîne les étapes suivantes pour NIRPS ou SPIROU :
  1. rsync     : synchronisation des données (sync_NIRPS ou sync_SPIROU)
  2. slinky    : raffinement de la solution de longueur d'onde (slinky_tools.run_slinky)
  3. residuals : calcul des cartes de correction empirique (residuals.py)
  4. telluric  : correction tellurique par objet (predict_abso.main)

Usage
-----
    python run_pipeline.py                   # toutes les étapes, instrument dans telluric_config.yaml
    python run_pipeline.py --skip-sync       # skip rsync
    python run_pipeline.py --skip-slinky     # skip slinky
    python run_pipeline.py --skip-residuals  # skip residuals
    python run_pipeline.py --only-telluric   # seulement la correction tellurique
    python run_pipeline.py --object PROXIMA  # un seul objet (sinon science_targets du YAML)
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
    """Affiche un message avec horodatage."""
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
    _step_header(f'ÉTAPE 1/4 — SYNC ({instrument})')
    sync_script = os.path.join(SCRIPT_DIR, f'sync_{instrument}')
    if not os.path.exists(sync_script):
        _tprint(f'Script {sync_script} introuvable — sync ignoré.')
        return
    ret = subprocess.call(['bash', sync_script], cwd=SCRIPT_DIR)
    if ret != 0:
        _tprint(f'AVERTISSEMENT : sync_{instrument} a retourné le code {ret}.')
    else:
        _tprint('Sync terminé.')


# ---------------------------------------------------------------------------
# Step 2 — slinky
# ---------------------------------------------------------------------------

def run_slinky():
    _step_header('ÉTAPE 2/4 — SLINKY (raffinement longueur d\'onde)')
    import slinky_tools
    slinky_tools.run_slinky()
    _tprint('Slinky terminé.')


# ---------------------------------------------------------------------------
# Step 3 — residuals
# ---------------------------------------------------------------------------

def run_residuals():
    _step_header('ÉTAPE 3/4 — RESIDUALS (cartes de correction empirique)')
    residuals_script = os.path.join(SCRIPT_DIR, 'residuals.py')
    ret = subprocess.call([sys.executable, residuals_script], cwd=SCRIPT_DIR)
    if ret != 0:
        _tprint(f'AVERTISSEMENT : residuals.py a retourné le code {ret}.')
    else:
        _tprint('Residuals terminé.')


# ---------------------------------------------------------------------------
# Step 4 — predict_abso par objet
# ---------------------------------------------------------------------------

def run_telluric(objects, instrument, batch_name, template_style, force_recompute):
    _step_header(f'ÉTAPE 4/4 — CORRECTION TELLURIQUE ({instrument})')
    _tprint(f'Objets : {objects}')

    # Import ici pour éviter le long chargement si on --list-objects seulement
    import predict_abso as pa

    first = True
    for obj in objects:
        _tprint(f'--- Traitement de {obj} ---')
        pa.main(
            batch_name=batch_name,
            instrument=instrument,
            obj=obj,
            template_style=template_style,
            force_recompute=(force_recompute and first),
        )
        first = False

    _tprint('Correction tellurique terminée.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = _load_telluric_config()
    default_instrument = cfg.get('instrument', 'NIRPS').upper()
    default_targets = cfg.get('science_targets', [])
    default_template = cfg.get('template_style', 'model')

    # Lire le nom du batch depuis batch_config.yaml
    batch_cfg_path = os.path.join(SCRIPT_DIR, 'batch_config.yaml')
    default_batch = 'skypca_v5'
    if os.path.exists(batch_cfg_path):
        with open(batch_cfg_path, 'r') as fh:
            batch_yaml = yaml.safe_load(fh)
        batch_section = batch_yaml.get('batch', {})
        default_batch = (batch_section.get('name') if isinstance(batch_section, dict)
                         else batch_yaml.get('batch_name', default_batch))

    parser = argparse.ArgumentParser(
        description='Pipeline complet de correction tellurique (rsync → slinky → residuals → telluric)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--instrument', default=default_instrument,
                        choices=['NIRPS', 'SPIROU'],
                        help='Instrument')
    parser.add_argument('--object', default=None,
                        help='Traiter un seul objet (sinon science_targets du YAML)')
    parser.add_argument('--batch', default=default_batch,
                        help='Nom du batch (identifiant de sortie)')
    parser.add_argument('--template', default=default_template,
                        choices=['model', 'self'],
                        help='Style de template stellaire')

    # Contrôle des étapes
    step_group = parser.add_argument_group('Contrôle des étapes')
    step_group.add_argument('--skip-sync',      action='store_true', help='Ne pas exécuter le rsync')
    step_group.add_argument('--skip-slinky',    action='store_true', help='Ne pas exécuter slinky')
    step_group.add_argument('--skip-residuals', action='store_true', help='Ne pas exécuter residuals')
    step_group.add_argument('--skip-telluric',  action='store_true', help='Ne pas exécuter la correction tellurique')
    step_group.add_argument('--only-telluric',  action='store_true',
                            help='Seulement la correction tellurique (implique --skip-sync --skip-slinky --skip-residuals)')

    parser.add_argument('--recompute', action='store_true',
                        help='Forcer le recalcul de la grille d\'absorption pré-calculée')

    args = parser.parse_args()

    if args.only_telluric:
        args.skip_sync = True
        args.skip_slinky = True
        args.skip_residuals = True

    instrument = args.instrument.upper()
    objects = [args.object] if args.object else default_targets

    if not objects:
        print('ERREUR : aucun objet spécifié et science_targets vide dans telluric_config.yaml.')
        sys.exit(1)

    t0 = time.time()
    _tprint(f'Pipeline démarré | instrument={instrument} | objets={objects} | batch={args.batch}')

    # ---- Étape 1 ----
    if not args.skip_sync:
        run_sync(instrument)

    # ---- Étape 2 ----
    if not args.skip_slinky:
        run_slinky()

    # ---- Étape 3 ----
    if not args.skip_residuals:
        run_residuals()

    # ---- Étape 4 ----
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
    _step_header(f'PIPELINE TERMINÉ — durée totale {h:02d}:{m:02d}:{s:02d}')


if __name__ == '__main__':
    main()
