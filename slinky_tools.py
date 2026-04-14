"""
Slinky wavelength solution refinement tools (adapté pour telluric_fit)

Ce module contient les fonctions principales du slinky, adaptées pour utiliser le YAML local et recevoir les données scientifiques en argument.
"""

import numpy as np
import os
import glob
from astropy.io import fits
from astropy.table import Table
from scipy.constants import c
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from tqdm import tqdm
import matplotlib.pyplot as plt

# Utiliser le loader YAML local (telluric_fit)
import yaml

def load_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        params = yaml.safe_load(file)
    return params

# --- Copier ici les fonctions principales du slinky (refine_wavesol, padding_wavesol, wrap, etc.) ---
# Pour l'instant, on pose un squelette minimal, puis on complète avec la logique copiée/adaptée.


def _find_fp_hc_files(params, scientific_data=None):
    """
    Détecte les fichiers FP et HC selon l'instrument ou les récupère depuis scientific_data.
    Retourne deux listes : fp_files, hc_files
    """
    if scientific_data is not None:
        # Si explicitement fourni, on attend un dict {'fp_files': [...], 'hc_files': [...]}
        fp_files = scientific_data.get('fp_files', [])
        hc_files = scientific_data.get('hc_files', [])
        if fp_files and hc_files:
            return fp_files, hc_files

    # Sinon, détection automatique selon l'instrument
    instrument = params.get('instrument', 'NIRPS').upper()
    if instrument == 'NIRPS':
        calib_dir = 'calib_NIRPS'
    elif instrument == 'SPIROU':
        calib_dir = 'calib_SPIROU'
    else:
        raise ValueError(f"Instrument inconnu: {instrument}")

    fp_files = sorted(glob.glob(os.path.join(calib_dir, '*wave_fplines_A.fits')))
    hc_files = sorted(glob.glob(os.path.join(calib_dir, '*wave_hclines_A.fits')))

    if not fp_files:
        print(f"[SLINKY] Aucun fichier FP trouvé dans {calib_dir}.")
    if not hc_files:
        print(f"[SLINKY] Aucun fichier HC trouvé dans {calib_dir}.")
    if not fp_files or not hc_files:
        raise FileNotFoundError(f"Fichiers FP/HC manquants pour l'instrument {instrument} dans {calib_dir}.")

    return fp_files, hc_files

def refine_wavesol(params, scientific_data=None):
    """
    Affiner la solution en longueur d'onde (slinky) en utilisant les paramètres YAML et les données scientifiques fournies.
    """
    fp_files, hc_files = _find_fp_hc_files(params, scientific_data)
    print(f"[SLINKY] {len(fp_files)} fichiers FP et {len(hc_files)} fichiers HC détectés.")
    # TODO: Insérer ici la logique de traitement réelle (voir slinky.py)
    # ...


def padding_wavesol(params, scientific_data=None):
    """
    Ajouter la solution en longueur d'onde raffinée aux fichiers FITS, en utilisant les paramètres YAML et les données scientifiques fournies.
    """
    fp_files, hc_files = _find_fp_hc_files(params, scientific_data)
    print(f"[SLINKY] (padding) {len(fp_files)} fichiers FP et {len(hc_files)} fichiers HC détectés.")
    # TODO: Insérer ici la logique de padding réelle (voir slinky.py)
    # ...

def run_slinky_from_yaml(yaml_path, scientific_data=None):
    """
    Point d'entrée principal : charge la config YAML et lance le slinky.
    """
    params = load_yaml(yaml_path)
    refine_wavesol(params, scientific_data=scientific_data)
    padding_wavesol(params, scientific_data=scientific_data)

# TODO: Ajouter d'autres fonctions utilitaires du slinky si besoin
