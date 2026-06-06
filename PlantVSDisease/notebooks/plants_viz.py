"""
Plant Vs Disease — Module de visualisation pour l'exploration du DataFrame.

Utilisé par df_plants_stats.ipynb
Les fonctions utilisent la variable globale `df` du module.
Avant d'appeler les fonctions, le notebook doit faire :
    import plants_viz
    plants_viz.df = <votre_dataframe>
"""

import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Variable globale du module — le notebook doit l'assigner
df = None


def display_diseases_histo():
    """Bar chart des effectifs de chaque maladie (hors 'healthy')."""
    diseases_without_healthy = df[df['label_disease'] != 'healthy']['label_disease'].unique()
    diseases_without_healthy_count = df[df['label_disease'] != 'healthy']['label_disease'].value_counts()
    plt.figure(figsize=(25, 10))
    plt.bar(diseases_without_healthy, diseases_without_healthy_count, width=0.6, label="Maladies")
    plt.xticks(rotation=45)
    plt.title("Effectifs des maladies (toutes images)")


def display_cat_images():
    """Affiche les statistiques sur les catégories d'images."""
    num_images = df.shape[0]
    num_cat_images = pd.DataFrame(df.groupby(['cat_num_image'])['filename']).shape[0]
    num_mean_images_by_cat = df.groupby(['cat_num_image']).agg({"label_all": pd.Series.nunique}).mean()
    num_mean_images_by_cat = re.findall(r'\d+', str(num_mean_images_by_cat))
    num_mean_images_by_cat = num_mean_images_by_cat[0] + '.' + num_mean_images_by_cat[1]
    num_max_images_by_cat = df.groupby(['cat_num_image']).agg({"label_all": pd.Series.nunique}).max()
    num_max_images_by_cat = re.findall(r'\d+', str(num_max_images_by_cat))
    num_max_images_by_cat = num_max_images_by_cat[0]
    num_min_images_by_cat = df.groupby(['cat_num_image']).agg({"label_all": pd.Series.nunique}).min().astype(int)
    num_min_images_by_cat = re.findall(r'\d+', str(num_min_images_by_cat))
    num_min_images_by_cat = num_min_images_by_cat[0]

    print("Nombre d'images: ", num_images)
    print("Nombre de categories d'images: ", num_cat_images)
    print("Nombre moyen d'images par categorie: ", num_mean_images_by_cat)
    print("Nombre maximum d'images dans une categorie: ", num_max_images_by_cat)
    print("Nombre minimum d'images dans une categorie: ", num_min_images_by_cat)


def display_images_cat_box():
    """Box plot du nombre d'images par catégorie."""
    df1 = df.groupby(['cat_num_image']).agg({"label_all": pd.Series.nunique})
    plt.figure(figsize=(10, 10))
    plt.boxplot(df1, tick_labels=["Nombre d'images par categorie"])
    plt.title("Box plot du nombre d'images par catégories")


def display_size_images():
    """Affiche les statistiques sur la taille des fichiers images."""
    mean_images_size = round(df['file_size_meg'].mean(), 3)
    max_images_size = df['file_size_meg'].max()
    min_images_size = df['file_size_meg'].min()
    median_images_size = round(df['file_size_meg'].median(), 3)

    print("Taille moyenne des images (Meg): ", mean_images_size)
    print("Taille mediane des images: ", median_images_size)
    print("Taille maximum des images: ", max_images_size)
    print("Taille minimum des images: ", min_images_size)


def display_size_images_cat_box():
    """Box plot de la taille des images."""
    plt.figure(figsize=(10, 10))
    plt.boxplot(df['file_size_meg'], tick_labels=['Taille des images'])
    plt.title("Box plot de la taille des images")


def display_some_images_from_dataframe():
    """Affiche les 8 premières images en niveaux de gris depuis les fichiers."""
    sample_paths = df['filepath'].head(8).values
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), subplot_kw=dict(xticks=[], yticks=[]))
    for i, ax in enumerate(axes.flat):
        img = Image.open(sample_paths[i]).convert('L').resize((62, 47))
        ax.imshow(np.array(img), cmap='gray')
    plt.tight_layout()