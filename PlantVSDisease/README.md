# PlantVsDisease

Projet **Datascientest** — Bootcamp Data Science **Sep 2022**

Classification d'images de plantes malades / saines à partir du [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (Kaggle).

---

## Équipe

| Rôle | Nom |
|------|-----|
| Apprenant | Bertrand Chatelet |
| Apprenant | [François Condominas](https://www.linkedin.com/in/francoiscondominas) — [GitHub](https://github.com/fcondo56) |
| Apprenant | Jeremy Cuvelier |
| Mentor | Zakary Saheb |
| Chef de cohorte | Romain Godet |

---

## Structure du projet

```
PlantVSDisease/
├── README.md
├── data/
│   ├── PlantVSDiseasesDataset/
│   │   ├── train/          # 38 sous-dossiers (plante___maladie)
│   │   └── valid/          # Sous-ensemble de validation
│   └── test/
├── notebooks/
│   ├── df_plant_new2.csv                   # DataFrame exporté
│   ├── DataExplore_MAIN_TeamShared_2810.ipynb
│   ├── DataExplore_MAIN_TeamShared_0211.ipynb
│   ├── DataExplore_MAIN_TeamShared_0811_transform_images.ipynb
│   ├── df_plants_stats.ipynb
│   ├── plant_utils.py                      # Module utilitaire (validation)
│   ├── plant_transform.py                  # Module de transformation d'images
│   └── plants_viz.py                       # Module de visualisation
└── venv/                                   # Environnement virtuel Python
```

---

## Notebooks

### 1. `DataExplore_MAIN_TeamShared_2810.ipynb`

**Version historique de référence** — première exploration du dataset.

- Import des packages
- Construction du DataFrame à partir des sous-dossiers d'images via `raw_data_to_dataframe()`
- Export du DataFrame en CSV (`df_plant_new2.csv`)
- Débogage et tests unitaires de fonctions
- Première architecture de modèle séquentiel (2 couches卷积, Dense finale 38 classes)
- Entraînement et évaluation du modèle

### 2. `DataExplore_MAIN_TeamShared_0211.ipynb`

**Version allégée et modernisée** — entraînement avec DataGenerator.

- Mêmes imports et construction du DataFrame que la version 2810
- **LabelEncoder** binaire : `healthy` / `Nonehealthy` (au lieu des 38 classes)
- **DataGenerator** (`df_to_generator`) : charge les images à la volée depuis les `filepath` pour économiser la mémoire RAM
- **Modèle CNN** : 2 couches `Conv2D(32, 3×3) + MaxPooling2D + Dropout(0.2)`, sortie Dense(2, softmax)
- Entraînement avec `model.fit(train_gen(), ...)` sans charger toutes les images en mémoire

### 3. `DataExplore_MAIN_TeamShared_0811_transform_images.ipynb`

**Version avec rehaussement d'images et sous-échantillonnage.**

- Utilise `plant_transform.py` pour charger les images AVEC prétraitement (`images_enhancement`)
- 6 modes de rehaussement testés : `RGB`, `RGB-HE`, `L`, `L-HE`, `L-LHE`, `L-CLAHE`
- LabelEncoder binaire (`healthy` / `Nonehealthy`)
- **Undersampling** avec `RandomUnderSampler` (librairie `imbalanced-learn`) pour équilibrer les classes
- Modèle CNN identique à la version 0211
- **EarlyStopping** (patience=5) pour éviter le surapprentissage
- Comparaison des performances F1-score selon les modes de rehaussement :
  - `RGB` → accuracy ≈ 0.68 (meilleur résultat)
  - `RGB-HE` → accuracy ≈ 0.68 (amélioration de contraste)
  - `L` → accuracy ≈ 0.57
  - `L-CLAHE` → accuracy ≈ 0.60

### 4. `df_plants_stats.ipynb`

**Exploration statistique du DataFrame** (analyse descriptive).

- Chargement du CSV exporté
- Histogramme des effectifs par maladie
- Statistiques sur les catégories d'images (nombre, moyenne, min, max)
- Box plot du nombre d'images par catégorie
- Statistiques sur les tailles de fichiers (moyenne, médiane, min, max)
- Box plot de la distribution des tailles
- Affichage d'échantillons visuels (8 premières images en niveaux de gris)

---

## Scripts Python

### `plant_utils.py`

Module utilitaire de base, réutilisé par les notebooks 2810 et 0211.

| Fonction | Rôle |
|----------|------|
| `start_flag()` | Affiche une bannière de démarrage avec l'heure |
| `end_flag()` | Affiche une bannière de fin |
| `manage_CWD()` | Définit le chemin racine des données (variable d'env `Images` ou chemin relatif) |
| `manage_CWD_arg(Images)` | Variante avec argument explicite |
| `raw_data_to_dataframe(data_path)` | Construit le DataFrame en scannant les sous-dossiers, filtre les images invalides |
| `df_to_csv(df, name_csv)` | Exporte le DataFrame en CSV |
| `df_to_generator(df, indices, labels, batch_size, target_size)` | DataGenerator lisant les images à la volée avec OpenCV → RGB → resize → normalisation |

### `plant_transform.py`

Module de transformation et rehaussement d'images pour le notebook 0811.

| Fonction | Rôle |
|----------|------|
| `raw_data_with_images_to_dataframe(data_path, mode)` | Construit le DataFrame avec chargement des images EN MÉMOIRE et rehaussement |
| `images_enhancement(images, width, height, mode)` | Redimensionne et convertit les images selon le mode choisi (RGB, niveaux de gris, histogram equalization, CLAHE, etc.) |

### `plants_viz.py`

Module de visualisation pour le notebook `df_plants_stats.ipynb`.

| Fonction | Rôle |
|----------|------|
| `display_diseases_histo()` | Diagramme en barres des effectifs par maladie |
| `display_cat_images()` | Statistiques descriptives des catégories d'images |
| `display_images_cat_box()` | Box plot du nombre d'images par catégorie |
| `display_size_images()` | Statistiques sur les tailles de fichiers |
| `display_size_images_cat_box()` | Box plot de la taille des images |
| `display_some_images_from_dataframe()` | Affichage des 8 premières images en gris |

---

## Pipeline général

```
Kaggle Dataset (train/)
    │
    ▼
raw_data_to_dataframe()      ──> DataFrame (label_all, label_plant, label_disease, filepath)
    │
    ├── Export CSV ──> df_plant_new2.csv
    │
    ▼
LabelEncoder (healthy / Nonehealthy)  ──> y (binaire)
    │
    ▼
Train / Test split (80/20, stratifié)
    │
    ├── 2810 / 0211 : DataGenerator à la volée ──> model.fit(train_gen())
    │
    └── 0811 : Images en mémoire + Undersampling ──> model.fit(X_train, y_train)
    │
    ▼
Modèle CNN : Conv2D → MaxPooling → Dropout → Conv2D → MaxPooling → Dropout → Flatten → Dense(2, softmax)
    │
    ▼
Évaluation : classification_report (F1-score, précision, rappel)
```

---

## Dépendances principales

- Python ≥ 3.10
- TensorFlow / Keras
- OpenCV (`opencv-contrib-python`)
- scikit-learn, scikit-image
- pandas, numpy, matplotlib, seaborn, plotly
- imbalanced-learn (pour `RandomUnderSampler`)
- joblib, PIL, glob

Voir l'environnement virtuel dans le dossier `venv/`.