"""
Plant Vs Disease — Module de transformation d'images.

Fonctions spécifiques pour charger les images avec prétraitement
et rehaussement (enhancement). Utilisé par le notebook
DataExplore_MAIN_TeamShared_0811_transform_images.ipynb
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import cv2 as cv
from skimage import io, color, exposure, transform
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters import rank
import fidle


def raw_data_with_images_to_dataframe(data_path, mode):
    """
    Build a DataFrame from image folders, with enhancement.

    Loads images, filters out those < 0.013 MB,
    then applies images_enhancement.
    Returns a DataFrame with columns: label_all, label_plant,
    label_disease, filename, image.
    """
    print('data_path:', data_path)
    print('\n--------------------\n * RAW DATA TO DF * \n--------------------\n')

    mysubfolders = os.listdir(data_path)

    data_dico = {
        'label_all': [],
        'label_plant': [],
        'label_disease': [],
        'filename': [],
        'image': [],
    }

    dataNA_dico = {'label_all': []}

    for subdir in mysubfolders:
        if subdir in ['desktop.ini']:
            print('--- NA -----', subdir, '---')
            continue

        print(' --- now computing ---------------------------------', subdir, '---')
        folder_path = os.path.normpath(os.path.join(data_path, subdir))

        for file in os.listdir(folder_path):
            if file[-4:] not in {'.png', '.PNG', '.jpg', '.JPG'}:
                continue

            file_path = os.path.normpath(os.path.join(folder_path, file))
            im = cv.imread(file_path)

            if im is None:
                dataNA_dico['label_all'].append(subdir)
            else:
                file_with_path = folder_path + '/' + file
                file_stats = os.stat(file_with_path)
                file_size_meg = round(file_stats.st_size / (1024 * 1024), 3)
                if file_size_meg < 0.013:
                    continue

                im = im[:, :, ::-1]  # BGR → RGB
                data_dico['label_all'].append(subdir)
                data_dico['label_plant'].append(subdir[0: subdir.find('___')])
                data_dico['label_disease'].append(subdir[subdir.find('___') + 3:])
                data_dico['filename'].append(file)
                data_dico['image'].append(im)

    # NA image count
    print('\n------------------------\n * data/image removed * \n------------------------\n')
    print('Amout data/image removed from the RAW dataset:', len(dataNA_dico['label_all']))
    count_NA = Counter(dataNA_dico['label_all'])
    print('\nNA files per Label_all count:\n', count_NA)

    data_df = pd.DataFrame(data_dico)

    print('data_df["image"].shape:', data_df["image"].shape)
    data_df["image"] = images_enhancement(data_df["image"], 256, 256, mode)

    print('\nAmout data/image read/computed with success:', len(data_df.label_all))
    print('\ndf features:', data_df.columns)

    print('\n--------------------\n * data_df issued * \n--------------------\n')
    print('does not include image but image_path!')

    return data_df


def images_enhancement(images, width=256, height=256, mode='RGB'):
    """
    Resize and convert images - doesn't change originals.

    Input images must be RGBA or RGB.
    Note: all outputs are fixed size numpy array of float64.

    Args:
        images:         images list
        width, height:  new images size
        mode:           RGB | RGB-HE | L | L-HE | L-LHE | L-CLAHE

    Returns:
        numpy array of enhanced images as uint8 list
    """
    modes = {'RGB': 3, 'RGB-HE': 3, 'L': 1, 'L-HE': 1, 'L-LHE': 1, 'L-CLAHE': 1}
    lz = modes[mode]
    print("len(images):", len(images))
    out = []

    for img in images:
        # ---- if RGBA, convert to RGB
        if img.shape[2] == 4:
            img = color.rgba2rgb(img)

        # ---- Resize
        img = transform.resize(img, (width, height))

        # ---- RGB / Histogram Equalization
        if mode == 'RGB-HE':
            hsv = color.rgb2hsv(img.reshape(width, height, 3))
            hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
            img = color.hsv2rgb(hsv)

        # ---- Grayscale
        if mode == 'L':
            img = color.rgb2gray(img)

        # ---- Grayscale / Histogram Equalization
        if mode == 'L-HE':
            img = color.rgb2gray(img)
            img = exposure.equalize_hist(img)

        # ---- Grayscale / Local Histogram Equalization
        if mode == 'L-LHE':
            img = color.rgb2gray(img)
            img = img_as_ubyte(img)
            img = rank.equalize(img, disk(10)) / 255.

        # ---- Grayscale / Contrast Limited Adaptive Histogram Equalization (CLAHE)
        if mode == 'L-CLAHE':
            img = color.rgb2gray(img)
            img = exposure.equalize_adapthist(img)

        out.append(img)
        fidle.utils.update_progress('Enhancement: ', len(out), len(images))

    out = np.array(out, dtype='uint8')
    return out.tolist()