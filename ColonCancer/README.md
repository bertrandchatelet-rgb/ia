# RNA Cancer Stage Classification (PRJNA755688-stage12)

Classification binaire **control vs stade II** à partir de données d'expression de RNAs (séquençage RNA) — projet PRJNA755688-stage12.

## 📋 Notebooks

### `generate5-ml.ipynb` — RandomForest avec validation croisée 5-fold

Pipeline léger et fiable avec un **RandomForest régularisé** (`max_depth=10`, `min_samples_leaf=5`) et une validation croisée **StratifiedKFold 5-fold**.

#### Étapes

1. **Chargement** des données RNA (2653 features, 960 échantillons) depuis `rna/results-*.txt`
2. **Split 80/20** stratifié + **validation croisée 5-fold** avec pipeline complet dans chaque fold (undersampling → filtrage des RNAs → NCA + PCA → RandomForest)
3. **Entraînement final** sur tout le train + évaluation sur le test réservé
4. **Récapitulatif** complet (CV, test, matrice de confusion, diagnostic overfitting)

#### Résultats

| Métrique | CV 5-fold (moy ± σ) | Test final |
|----------|-------------------|------------|
| **Accuracy** | **0.9934 ± 0.0056** | **0.9868** |
| **Balanced Acc** | 0.9915 ± 0.0115 | 0.9750 |
| **MCC** | 0.9831 ± 0.0144 | 0.9661 |
| **Sensibilité** (control) | — | 0.9500 |
| **Spécificité** (II) | — | **1.0000** |

- Matrice de confusion (test) : `[[76, 4], [0, 223]]`
- Écart CV vs Test : **0.0066** ✅ stable et généralisable
- **Aucun data leakage** : split avant undersampling, sélection des features dans la boucle de CV

---

### `generate6-ml.ipynb` — AutoGluon (3 configurations)

Pipeline utilisant **AutoGluon** pour comparer différentes configurations (PCA, sans PCA, best_quality).

#### Résultats

| Configuration | Accuracy | Balanced Acc | MCC | Matrice de confusion |
|--------------|----------|-------------|-----|---------------------|
| **Avec PCA** (Cell 8) | 0.9922 | 0.9921 | 0.9845 | — |
| **Sans PCA** (Cell 9) | **0.9967** | **0.9978** | **0.9916** | — |
| **best_quality** (Cell 10) | 0.9934 | 0.9955 | 0.9833 | `[[80, 0], [2, 221]]` |

- **Meilleur modèle** : WeightedEnsemble_L2 (sans PCA, medium_quality)
- **Feature la plus importante** : `hsa-miR-6131` (importance 0.387)
- **Limitation** : la configuration `best_quality` nécessite >4 Go de RAM supplémentaires pour fonctionner sans erreur OOM

---

## 🧪 Comparaison des approches

| Critère | generate5 (RF + CV) | generate6 (AutoGluon) |
|---------|-------------------|---------------------|
| Modèle | RandomForest régularisé | AutoGluon (ensemble) |
| Validation | **5-fold CV** (robuste) | Split unique (simple) |
| Robustesse | ✅ Haute (CV + régularisation) | ⚠️ Modérée (pas de CV) |
| Performance test | 0.9868 | **0.9967** |
| Temps d'exécution | ~30s | ~5-10 min (best_quality ≈ 7 min) |

> **Note** : Les scores parfaits (1.0) observés dans les premières versions de ces notebooks étaient dus à un **sur-apprentissage massif** (data leakage, absence de CV, modèles non régularisés). Les résultats actuels sont biologiquement crédibles et statistiquement robustes.

## 📊 Données

- **Projet** : PRJNA755688-stage12
- **Échantillons** : 960 (561 control, 399 stade II)
  - Train : 1211 (dont 892 control, 319 II) — après split 80/20
  - Test : 303 (dont 223 control, 80 II) — jamais utilisé pendant la CV
- **Features** : 2653 RNAs (comptages d'expression)
- **Filtrage** : RNAs exprimés dans ≥20% des échantillons → top 50 par variance

## 🚀 Exécution

```bash
# Activer l'environnement
source .venv/bin/activate

# Lancer Jupyter
jupyter notebook generate5-ml.ipynb
# ou
jupyter notebook generate6-ml.ipynb
```

Dépendances principales : `pandas`, `scikit-learn`, `autogluon` (pour generate6), `numpy`