# Analyse et Améliorations de predict_abso.py

**Date**: 2026-01-12
**Analyste**: Claude Code
**Fichier analysé**: `predict_abso.py`

---

## 📊 Résumé Exécutif

Le code `predict_abso.py` effectue une correction tellurique sophistiquée sur des spectres astronomiques. L'analyse révèle un code fonctionnel mais qui bénéficierait grandement d'une refactorisation pour améliorer la maintenabilité, la reproductibilité et la flexibilité.

**Score de qualité global**: 6.5/10

**Points forts**:
- ✅ Algorithme solide et bien pensé
- ✅ Intégration efficace avec `tellu_tools`
- ✅ Gestion appropriée des métadonnées FITS

**Points à améliorer**:
- ❌ Configuration hardcodée
- ❌ Manque de modularité
- ❌ Documentation limitée
- ❌ Absence de gestion d'erreurs robuste

---

## 🔍 Analyse Détaillée

### 1. Structure et Organisation

#### Problèmes Identifiés

**1.1 Configuration hardcodée (lignes 23-27)**
```python
instrument = 'NIRPS'
obj = 'TOI4552'
batchname = 'skypca_v5'
template_style = 'model'
```

**Impact**:
- ⚠️ Nécessite de modifier le code pour chaque nouveau batch
- ⚠️ Risque d'erreurs lors de modifications manuelles
- ⚠️ Difficile de tracer quelle configuration a été utilisée

**Solution proposée**: Système de configuration externe (voir `predict_abso_config.py`)

---

**1.2 Code mort et blocs debug**

Plusieurs blocs `if False:` présents :
- Lignes 138-147 : Visualisation debug
- Lignes 342-374 : Comparaison DRS/APERO
- Lignes 293-295 : Masque O2 commenté

**Impact**:
- 📉 Réduit la lisibilité
- 🐛 Peut prêter à confusion
- 📦 Augmente la taille du code inutilement

**Solution**: Déplacer vers des scripts de visualisation séparés ou supprimer

---

**1.3 Imports redondants**

```python
from astropy.table import Table  # ligne 6
from astropy.table import Table  # ligne 9 (doublon)
```

**Solution**: Nettoyer les imports

---

### 2. Magic Numbers et Constantes

#### Valeurs non documentées

| Ligne | Valeur | Usage | Recommandation |
|-------|--------|-------|----------------|
| 44, 199 | `101` | Taille filtre passe-bas | Déplacer dans config |
| 225 | `0.1` | Seuil ratio valide | Nommer comme constante |
| 229, 230 | `3`, `0.3` | Seuils outliers ratio | Nommer comme constante |
| 231 | `501` | Fenêtre lissage | Déplacer dans config |
| 239 | `0.2` | Seuil flux bas | Nommer comme constante |
| 269 | `1` | Seuil rejet ciel | Nommer comme constante |

**Solution**: Toutes ces valeurs sont maintenant dans `predict_abso_config.py` avec documentation.

---

### 3. Gestion des Erreurs

#### Problèmes

**3.1 Absence de try-except**

```python
sp = fits.getdata(file)  # ligne 115
wave = fits.getdata(...)  # ligne 117
```

**Risque**: Crash complet si fichier corrompu ou manquant

**3.2 Vérifications minimales**

- Pas de validation des dimensions des tableaux
- Pas de vérification de cohérence wave/sp
- Pas de gestion des cas limites (tous NaN, etc.)

**Solution proposée**:

```python
try:
    sp = fits.getdata(file)
    if sp.shape != expected_shape:
        raise ValueError(f"Invalid spectrum shape: {sp.shape}")
except Exception as e:
    logger.error(f"Failed to load {file}: {e}")
    continue
```

---

### 4. Performance et Optimisation

#### Opportunités d'amélioration

**4.1 Pré-calcul des absorptions**

✅ Déjà bien fait : `all_abso` est pré-calculé (ligne 191)

**4.2 Calculs redondants**

- `np.nanpercentile` appelé plusieurs fois dans les boucles de visualisation
- `mp.lowpassfilter` pourrait être optimisé avec numba

**4.3 Parallélisation**

Le code traite les fichiers séquentiellement. Opportunité de parallélisation :

```python
from multiprocessing import Pool

with Pool(n_cores) as pool:
    results = pool.map(process_file, files)
```

---

### 5. Documentation

#### État actuel

- ❌ Pas de docstring de module
- ❌ Pas de docstrings pour le workflow principal
- ⚠️ Commentaires limités
- ❌ Pas de documentation des paramètres critiques

#### Améliorations apportées

La version refactorisée inclut :
- ✅ Docstring de module complet
- ✅ Docstrings pour toutes les fonctions
- ✅ Type hints pour les paramètres
- ✅ Commentaires expliquant la logique

---

## 🚀 Solutions Proposées

### Solution 1: Système de Configuration (predict_abso_config.py)

**Avantages**:
- ✅ Configuration centralisée
- ✅ Validation des paramètres
- ✅ Facilite les batchs multiples
- ✅ Traçabilité améliorée

**Usage**:
```python
from predict_abso_config import get_batch_config

config = get_batch_config(
    batch_name='skypca_v5',
    instrument='NIRPS',
    obj='TOI4552',
    template_style='model'
)
```

---

### Solution 2: Code Refactorisé (predict_abso_refactored.py)

**Améliorations clés**:

1. **Modularité**: Fonctions bien définies avec responsabilités claires
2. **Documentation**: Docstrings complètes avec type hints
3. **Configuration**: Système de batch externe
4. **Interface CLI**: Arguments en ligne de commande
5. **Maintenabilité**: Code plus lisible et testable

**Nouvelle structure**:

```
predict_abso_refactored.py
├── load_template()           # Chargement template
├── initialize_residuals()    # Chargement corrections
├── compute_initial_exponents()  # Calcul exposants initiaux
├── clean_template_ratio()    # Nettoyage template
├── apply_post_correction()   # Correction empirique
├── save_corrected_spectrum() # Sauvegarde résultats
├── process_single_file()     # Pipeline complet pour 1 fichier
└── main()                    # Point d'entrée principal
```

---

### Solution 3: Interface Ligne de Commande

**Avant** (modification du code nécessaire):
```python
# Éditer les lignes 23-27
instrument = 'NIRPS'
obj = 'TOI4552'
```

**Après** (interface CLI):
```bash
# Traitement avec paramètres par défaut
python predict_abso_refactored.py

# Traitement personnalisé
python predict_abso_refactored.py \
    --instrument NIRPS \
    --object TOI4552 \
    --batch skypca_v5 \
    --template model

# Lister les objets disponibles
python predict_abso_refactored.py --list-objects --instrument NIRPS
```

---

## 📋 Recommandations d'Implémentation

### Phase 1: Migration (Court terme)

1. **Tester la version refactorisée** sur un sous-ensemble de données
   ```bash
   # Créer un répertoire de test
   cp predict_abso.py predict_abso_original.py
   cp predict_abso_refactored.py predict_abso.py

   # Tester sur 1-2 fichiers
   python predict_abso.py --object TOI4552
   ```

2. **Validation des résultats**
   - Comparer les spectres corrigés (ancien vs nouveau)
   - Vérifier les headers FITS
   - Comparer les exposants optimisés

3. **Ajustements si nécessaire**
   - Adapter les seuils si besoin
   - Affiner la documentation

### Phase 2: Améliorations (Moyen terme)

1. **Logging professionnel**
   ```python
   import logging

   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler(f'tellu_corr_{batch_name}.log'),
           logging.StreamHandler()
       ]
   )
   ```

2. **Tests unitaires**
   ```python
   # test_predict_abso.py
   def test_load_template():
       ...

   def test_compute_initial_exponents():
       ...
   ```

3. **Parallélisation**
   ```python
   from joblib import Parallel, delayed

   results = Parallel(n_jobs=4)(
       delayed(process_single_file)(file, config, ...)
       for file in files
   )
   ```

### Phase 3: Optimisation (Long terme)

1. **Base de données des résultats**
   - SQLite pour stocker métadonnées
   - Facilite les requêtes et analyses

2. **Pipeline automatisé**
   - Détection automatique de nouveaux fichiers
   - Traitement par batch automatique

3. **Dashboard de monitoring**
   - Suivi de la qualité des corrections
   - Visualisation des tendances temporelles

---

## 🔧 Guide d'Utilisation

### Utilisation Basique

```python
# Import
from predict_abso_refactored import main

# Exécution simple
main(
    batch_name='skypca_v5',
    instrument='NIRPS',
    obj='TOI4552',
    template_style='model'
)
```

### Utilisation Avancée avec Configuration Personnalisée

```python
from predict_abso_config import get_batch_config
from predict_abso_refactored import main

# Configuration personnalisée
config = get_batch_config('my_batch', 'NIRPS', 'TOI4552', 'model')

# Ajuster paramètres
config['lowpass_filter_size'] = 151
config['sky_rejection_threshold'] = 0.8

# Sauvegarder config pour traçabilité
import json
with open(f'config_{config["batch_name"]}.json', 'w') as f:
    json.dump(config, f, indent=2)

# Exécuter
main(**config)
```

### Traitement de Multiples Objets

```python
objects = ['TOI4552', 'TOI1234', 'HD189733']

for obj in objects:
    print(f"\n{'='*60}")
    print(f"Processing {obj}")
    print(f"{'='*60}\n")

    main(
        batch_name='skypca_v5',
        instrument='NIRPS',
        obj=obj,
        template_style='model'
    )
```

---

## 📊 Comparaison Avant/Après

| Critère | Avant | Après | Amélioration |
|---------|-------|-------|--------------|
| **Configuration** | Hardcodée | Externe | ⭐⭐⭐⭐⭐ |
| **Documentation** | Minimale | Complète | ⭐⭐⭐⭐⭐ |
| **Modularité** | Monolithique | Fonctions | ⭐⭐⭐⭐ |
| **Testabilité** | Difficile | Facile | ⭐⭐⭐⭐⭐ |
| **Interface** | Édition code | CLI | ⭐⭐⭐⭐⭐ |
| **Maintenabilité** | Moyenne | Excellente | ⭐⭐⭐⭐⭐ |
| **Traçabilité** | Faible | Forte | ⭐⭐⭐⭐ |
| **Gestion erreurs** | Minimale | Robuste | ⭐⭐⭐⭐ |

---

## 🐛 Bugs Potentiels Identifiés

### Bug 1: Division par zéro potentielle

**Localisation**: Ligne 222-223 (original)
```python
ratio = (sp_tmp[iord]/template2[iord])
```

**Risque**: Si `template2[iord]` contient des zéros

**Solution**: Déjà géré par NaN propagation, mais pourrait être explicite

---

### Bug 2: Index out of bounds potentiel

**Localisation**: Ligne 258 (original)
```python
abso_scaling[abso_case==1] = expos[0]
```

**Risque**: Si `expos` n'a pas la bonne longueur

**Solution**: Validation de la longueur de `expos`

---

## 📝 Notes Additionnelles

### Dépendances Critiques

Le code dépend fortement de `tellu_tools.py`. Améliorations futures :

1. **Versionning**: Ajouter numéro de version dans `tellu_tools`
2. **Tests de compatibilité**: Vérifier versions compatibles
3. **Documentation croisée**: Liens entre modules

### Performance

**Benchmarks suggérés**:
- Temps par fichier
- Utilisation mémoire
- Efficacité I/O

**Optimisations possibles**:
- Caching des templates
- Pré-chargement des calibrations
- Parallélisation des ordres spectraux

---

## ✅ Checklist de Migration

- [ ] Sauvegarder version originale (`predict_abso_original.py`)
- [ ] Copier `predict_abso_config.py` dans le répertoire
- [ ] Tester `predict_abso_refactored.py` sur données test
- [ ] Comparer résultats (ancien vs nouveau)
- [ ] Valider les headers FITS
- [ ] Vérifier les exposants optimisés
- [ ] Tester l'interface CLI
- [ ] Documenter les différences observées
- [ ] Obtenir validation scientifique
- [ ] Renommer `predict_abso_refactored.py` → `predict_abso.py`
- [ ] Archiver ancienne version
- [ ] Mettre à jour documentation projet

---

## 📚 Ressources Supplémentaires

### Documentation
- TAPAS: [http://tapas.aeris-data.fr/](http://tapas.aeris-data.fr/)
- Astropy FITS: [https://docs.astropy.org/en/stable/io/fits/](https://docs.astropy.org/en/stable/io/fits/)

### Outils suggérés
- **pytest**: Tests unitaires
- **black**: Formatage code
- **pylint**: Analyse qualité code
- **sphinx**: Génération documentation

---

## 🎯 Conclusion

Le code `predict_abso.py` est fonctionnel mais bénéficierait grandement de la refactorisation proposée. Les améliorations apportées augmentent significativement :

- La **maintenabilité** du code
- La **reproductibilité** des analyses
- La **flexibilité** pour nouveaux cas d'usage
- La **traçabilité** des traitements

**Recommandation**: Procéder à la migration progressive en validant soigneusement chaque étape.

---

**Prochaines étapes suggérées**:

1. ✅ Révision de ce document
2. ⏳ Tests sur données réelles
3. ⏳ Validation scientifique
4. ⏳ Migration complète
5. ⏳ Documentation utilisateur
6. ⏳ Formation équipe

