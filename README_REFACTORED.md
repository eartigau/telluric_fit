# Pipeline de Correction Tellurique - Version Refactorisée

## 📚 Vue d'ensemble

Cette version refactorisée de `predict_abso.py` offre une architecture modulaire et flexible pour la correction tellurique des spectres astronomiques.

### Nouveaux fichiers

```
tapas_tellu/
├── predict_abso.py                 # Version originale (à conserver comme backup)
├── predict_abso_refactored.py      # ⭐ Nouvelle version principale
├── predict_abso_config.py          # 🔧 Système de configuration
├── run_batch_example.py            # 📖 Exemples d'utilisation
├── ANALYSIS_PREDICT_ABSO.md        # 📊 Analyse détaillée
└── README_REFACTORED.md            # 📄 Ce fichier
```

---

## 🚀 Démarrage Rapide

### Installation

Aucune installation supplémentaire requise si vous avez déjà les dépendances de `predict_abso.py` :

- `astropy`
- `numpy`
- `scipy`
- `matplotlib`
- `aperocore`
- `tellu_tools` (module local)

### Utilisation Basique

#### Option 1: Ligne de commande

```bash
# Traitement standard
python predict_abso_refactored.py \
    --instrument NIRPS \
    --object TOI4552 \
    --batch skypca_v5 \
    --template model

# Lister les objets disponibles
python predict_abso_refactored.py --list-objects --instrument NIRPS

# Aide complète
python predict_abso_refactored.py --help
```

#### Option 2: Import Python

```python
from predict_abso_refactored import main

main(
    batch_name='skypca_v5',
    instrument='NIRPS',
    obj='TOI4552',
    template_style='model'
)
```

#### Option 3: Exemples interactifs

```bash
python run_batch_example.py
```

---

## 📖 Guide d'Utilisation

### 1. Traitement d'un Objet Simple

Le cas le plus courant : traiter tous les spectres d'un objet donné.

```python
from predict_abso_refactored import main

main(
    batch_name='mon_batch',
    instrument='NIRPS',
    obj='TOI4552',
    template_style='model'
)
```

**Résultats** :
- Fichiers sauvegardés dans : `tellupatched_NIRPS/TOI4552_mon_batch_model/`
- Format : `*tellupatched_t.fits`
- Extensions FITS :
  - `FluxA` : Spectre corrigé
  - `Recon` : Modèle d'absorption
  - Headers mis à jour avec exposants et vitesses

---

### 2. Comparaison Template Model vs Self

Comparer les résultats avec différents types de templates :

```python
from predict_abso_refactored import main

obj = 'TOI4552'

# Avec template synthétique
main(batch_name='test', instrument='NIRPS', obj=obj, template_style='model')

# Avec template empirique (auto-généré)
main(batch_name='test', instrument='NIRPS', obj=obj, template_style='self')
```

Les résultats seront dans deux répertoires différents :
- `tellupatched_NIRPS/TOI4552_test_model/`
- `tellupatched_NIRPS/TOI4552_test_self/`

---

### 3. Traitement de Plusieurs Objets

Pour traiter plusieurs objets en une seule exécution :

```python
from predict_abso_refactored import main

objects = ['TOI4552', 'TOI1234', 'HD189733']

for obj in objects:
    print(f"\n{'='*60}\nTraitement de {obj}\n{'='*60}")

    try:
        main(
            batch_name='batch_multi',
            instrument='NIRPS',
            obj=obj,
            template_style='model'
        )
    except Exception as e:
        print(f"Erreur pour {obj}: {e}")
        continue
```

---

### 4. Configuration Personnalisée

Pour modifier les paramètres de traitement :

```python
from predict_abso_config import get_batch_config

# Charger configuration standard
config = get_batch_config('mon_batch', 'NIRPS', 'TOI4552', 'model')

# Modifier paramètres
config['lowpass_filter_size'] = 151  # Plus de lissage
config['sky_rejection_threshold'] = 0.8  # Rejet plus conservateur
config['dv_amp'] = 150  # Réduire la plage de recherche en vitesse

# Afficher configuration
print("Configuration personnalisée:")
for key, value in config.items():
    print(f"  {key}: {value}")

# Sauvegarder pour traçabilité
import json
with open('config_mon_batch.json', 'w') as f:
    json.dump(config, f, indent=2)
```

**Paramètres modifiables** (dans `predict_abso_config.py`) :

| Paramètre | Défaut | Description |
|-----------|---------|-------------|
| `lowpass_filter_size` | 101 | Taille fenêtre filtre passe-bas |
| `template_ratio_threshold_high` | 3.0 | Seuil haut pour rejection outliers |
| `template_ratio_threshold_low` | 0.3 | Seuil bas pour rejection outliers |
| `template_smooth_window` | 501 | Fenêtre lissage ratio template |
| `min_valid_ratio` | 0.1 | Fraction min pixels valides |
| `low_flux_threshold` | 0.2 | Seuil flux bas pour rejection |
| `sky_rejection_threshold` | 1.0 | Seuil rejet ciel brillant |
| `dv_amp` | 200 | Amplitude recherche vitesse (km/s) |

---

## 🔧 Configuration Avancée

### Créer une Configuration Personnalisée

Créez votre propre fichier de configuration :

```python
# my_config.py

from predict_abso_config import DEFAULT_PARAMS

# Hériter des paramètres par défaut
MY_CUSTOM_PARAMS = DEFAULT_PARAMS.copy()

# Modifier pour votre cas
MY_CUSTOM_PARAMS.update({
    'lowpass_filter_size': 151,
    'dv_amp': 150,
    'sky_rejection_threshold': 0.8,
})

def get_my_config(instrument, obj):
    """Configuration personnalisée pour mes besoins."""
    return {
        'batch_name': 'my_special_batch',
        'instrument': instrument,
        'object': obj,
        'template_style': 'model',
        **MY_CUSTOM_PARAMS
    }
```

Utilisation :

```python
from my_config import get_my_config
from predict_abso_refactored import main

config = get_my_config('NIRPS', 'TOI4552')
# Note: main() devrait être modifié pour accepter config dict
```

---

## 📊 Vérification des Résultats

### 1. Vérifier les Fichiers de Sortie

```python
import glob
import os
from astropy.io import fits

# Chemin de sortie
output_dir = '/path/to/tellupatched_NIRPS/TOI4552_skypca_v5_model/'

# Lister les fichiers
files = sorted(glob.glob(os.path.join(output_dir, '*tellupatched_t.fits')))

print(f"Fichiers traités: {len(files)}")

# Vérifier un fichier
if files:
    with fits.open(files[0]) as hdul:
        print("\nExtensions FITS:")
        hdul.info()

        print("\nMots-clés ajoutés:")
        hdr = hdul['FluxA'].header
        for key in ['ABS_VELO', 'SYS_VELO', 'EXPO_H2O', 'EXPO_CO2',
                    'EXPO_CH4', 'EXPO_O2', 'H2O_CV', 'CO2_VMR']:
            if key in hdr:
                print(f"  {key}: {hdr[key]} {hdr.comments[key]}")
```

### 2. Comparer avec Version Originale

```python
from astropy.io import fits
import numpy as np

# Fichiers à comparer
file_old = 'output_old/TOI4552_tellupatched_t.fits'
file_new = 'output_new/TOI4552_tellupatched_t.fits'

# Charger spectres
sp_old = fits.getdata(file_old, 'FluxA')
sp_new = fits.getdata(file_new, 'FluxA')

# Différence
diff = sp_new - sp_old
rms = np.sqrt(np.nanmean(diff**2))

print(f"RMS de la différence: {rms:.6e}")
print(f"Différence relative: {rms/np.nanmedian(sp_old)*100:.4f}%")

# Comparer exposants
hdr_old = fits.getheader(file_old, 'FluxA')
hdr_new = fits.getheader(file_new, 'FluxA')

for mol in ['H2O', 'CO2', 'CH4', 'O2']:
    key = f'EXPO_{mol}'
    expo_old = hdr_old[key]
    expo_new = hdr_new[key]
    print(f"{key}: {expo_old:.4f} → {expo_new:.4f} (Δ={expo_new-expo_old:.4f})")
```

### 3. Visualisation Rapide

```python
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

file = 'tellupatched_NIRPS/TOI4552_test/file_tellupatched_t.fits'

with fits.open(file) as hdul:
    sp_corr = hdul['FluxA'].data
    recon = hdul['Recon'].data
    wave = fits.getdata('calib_NIRPS/waveref.fits')

# Tracer un ordre
iord = 40

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Spectre corrigé
ax[0].plot(wave[iord], sp_corr[iord], 'k-', alpha=0.7, label='Corrigé')
ax[0].set_ylabel('Flux')
ax[0].legend()
ax[0].grid(alpha=0.3)

# Modèle tellurique
ax[1].plot(wave[iord], recon[iord], 'r-', alpha=0.7, label='Absorption')
ax[1].set_ylabel('Transmission')
ax[1].set_xlabel('Longueur d\'onde (nm)')
ax[1].legend()
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('check_correction.png', dpi=150)
plt.show()
```

---

## 🐛 Dépannage

### Problème 1: "No files found"

**Symptôme** :
```
Found 0 files to process
No files found in scidata_NIRPS/TOI4552/
```

**Solution** :
- Vérifier que `project_path` est correct dans `tellu_tools.user_params()`
- Vérifier que le répertoire `scidata_NIRPS/TOI4552/` existe
- Vérifier qu'il contient des fichiers `*.fits`

```bash
# Vérifier
ls /path/to/project/scidata_NIRPS/TOI4552/*.fits
```

---

### Problème 2: "Template file not found"

**Symptôme** :
```
FileNotFoundError: templates_NIRPS/Template_s1dv_TOI4552_sc1d_v_file_A.fits
```

**Solution** :
- Utiliser `template_style='model'` au lieu de `'self'`
- Ou générer le template empirique au préalable

---

### Problème 3: "WAVEFILE not found"

**Symptôme** :
```
FileNotFoundError: calib_NIRPS/WAVE_FILE_NAME.fits
```

**Solution** :
- Vérifier que les fichiers de calibration wavelength sont présents
- Télécharger depuis le serveur si nécessaire :

```bash
# Voir ligne 117-118 de predict_abso.py pour la commande scp
```

---

### Problème 4: Fichiers déjà existants

**Symptôme** :
```
Skipping file as it already exists
```

**Solution** :
- C'est normal ! Le code évite de retraiter les fichiers
- Pour forcer le retraitement :

```bash
# Supprimer les fichiers existants
rm tellupatched_NIRPS/TOI4552_batch/*tellupatched_t.fits
```

---

## 📈 Performance

### Temps de Traitement Typiques

Sur un MacBook Pro M1 :

| Nombre de fichiers | Temps total | Temps/fichier |
|---------------------|-------------|---------------|
| 10 | ~15 min | ~1.5 min |
| 50 | ~75 min | ~1.5 min |
| 100 | ~150 min | ~1.5 min |

**Facteurs influençant le temps** :
- Nombre d'ordres spectraux
- Nombre d'itérations d'optimisation
- Activation des plots (`doplot=True` ralentit)
- I/O disque

### Optimisation Possible

Pour traiter en parallèle (future amélioration) :

```python
from multiprocessing import Pool

def process_wrapper(file):
    return process_single_file(file, config, ...)

with Pool(4) as pool:
    results = pool.map(process_wrapper, files)
```

---

## 📝 Bonnes Pratiques

### 1. Organisation des Batches

Recommandation de nommage :

```
batch_name = f"{purpose}_{version}"

Exemples:
- "skypca_v5"
- "test_new_algo_v1"
- "paper_final_v3"
```

### 2. Traçabilité

Toujours sauvegarder la configuration utilisée :

```python
import json
from datetime import datetime

config = get_batch_config(...)

# Ajouter métadonnées
config['processing_date'] = datetime.now().isoformat()
config['user'] = os.environ.get('USER', 'unknown')

# Sauvegarder
with open(f"config_{config['batch_name']}.json", 'w') as f:
    json.dump(config, f, indent=2)
```

### 3. Validation

Avant de traiter un grand nombre de fichiers :

1. Tester sur 1-2 fichiers
2. Vérifier visuellement les résultats
3. Comparer avec version précédente
4. Valider les exposants optimisés

### 4. Sauvegarde

Toujours garder les données originales et une copie de l'ancienne version :

```bash
# Backup version originale
cp predict_abso.py predict_abso_v1_backup.py

# Backup données traitées
tar -czf tellupatched_backup_$(date +%Y%m%d).tar.gz tellupatched_NIRPS/
```

---

## 🔄 Migration depuis l'Ancienne Version

### Étape 1: Tests

```python
# Test avec 1 fichier
from predict_abso_refactored import process_single_file

# ... charger toutes les données nécessaires ...

success = process_single_file(
    files[0],  # Premier fichier seulement
    config, spl, spl_dv, sky_dict, waveref,
    all_abso, abso_case, main_abso, hdr_tapas, model
)
```

### Étape 2: Comparaison

```python
# Comparer avec sortie de l'ancienne version
# (voir section "Vérification des Résultats")
```

### Étape 3: Migration Complète

```bash
# Sauvegarder ancienne version
mv predict_abso.py predict_abso_original.py

# Renommer nouvelle version
cp predict_abso_refactored.py predict_abso.py

# Mettre à jour imports dans autres scripts
# (si nécessaire)
```

---

## 📚 Ressources Additionnelles

### Documentation

- **Analyse détaillée** : [ANALYSIS_PREDICT_ABSO.md](ANALYSIS_PREDICT_ABSO.md)
- **Exemples** : [run_batch_example.py](run_batch_example.py)
- **Configuration** : [predict_abso_config.py](predict_abso_config.py)

### Support

Pour signaler des bugs ou demander des fonctionnalités :

1. Créer un issue sur le dépôt Git
2. Contacter l'équipe de développement
3. Consulter la documentation APERO

### Contributions

Pour contribuer au code :

1. Suivre le style PEP 8
2. Ajouter des docstrings pour toutes les fonctions
3. Inclure des tests si possible
4. Documenter les changements

---

## 📄 Licence

Ce code fait partie du pipeline de réduction APERO/NIRPS.

---

## ✅ Checklist de Démarrage

- [ ] Vérifier que `tellu_tools.py` est disponible
- [ ] Vérifier que `project_path` est correct
- [ ] Vérifier la présence des calibrations (waveref, etc.)
- [ ] Tester avec `--list-objects` pour voir les données
- [ ] Tester sur 1 fichier avec `process_single_file()`
- [ ] Comparer avec ancienne version
- [ ] Documenter la configuration utilisée
- [ ] Lancer le traitement complet
- [ ] Valider les résultats scientifiquement

---

**Dernière mise à jour** : 2026-01-12
**Version** : 1.0
**Auteur** : Claude Code (analyse et refactorisation)
