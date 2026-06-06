"""
Plant Vs Disease — Module de fonctions utilitaires validées.

Utilisé par DataExplore_MAIN_TeamShared_2810.ipynb
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
import cv2 as cv


def start_flag():
    """Print a start flag with current time."""
    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Plant Vs Disease  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print('...', current_time)


def end_flag():
    """Print an end flag."""
    print('\n------ Thats all folks! ------ )"~°(')


def manage_CWD():
    """Manage CWD & default folder path from 'Images' env variable.

    Falls back to a relative path if the env variable is not set.
    """
    print('\n---------\n * CWD * \n---------\n')
    data_path = os.getenv("Images")
    if data_path is None:
        data_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'PlantVSDiseasesDataset', 'train')
        )
        print('⚠️  Variable d\'environnement "Images" non définie.')
        print('   Utilisation du chemin relatif par défaut.')
    print('Current CWD:', data_path)
    return data_path


def manage_CWD_arg(Images):
    '''Manage CWD & default folder path'''

    # CWD
    print('\n---------\n * CWD * \n---------\n')

    # Chemin relatif par rapport au dossier du notebook
    data_path = os.path.normpath(os.path.join(os.getcwd(), '..', 'data', 'PlantVSDiseasesDataset', 'train'))
    print('Current CWD:', data_path)

    return(data_path)


def df_to_csv(df, name_csv):
    """
    Store df into csv at pointed location.

    Cannot overwrite a file.csv already there.
    INPUT: name_csv as name_csv.csv
    """
    print('\n---------------\n * DF to CSV * \n---------------\n')

    data_path_cwd = os.getcwd()
    path_to_file = '/' + name_csv + '.csv'
    path_to_file = os.path.normpath(path_to_file)
    path_to_file_all = os.path.normpath(data_path_cwd + path_to_file)

    df.to_csv(path_to_file_all,
              sep=',',
              index=False,
              header=True)

    print('csv store into:', path_to_file_all)


def df_to_generator(df, indices, labels, batch_size=32, target_size=(256, 256)):
    """
    DataGenerator that reads images from filepaths on the fly.

    Yields batches of (images, labels) for use with model.fit().
    Images are loaded via OpenCV, converted to RGB, resized, and normalized.
    """
    def gen():
        while True:
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_files = df.iloc[batch_idx]["filepath"].values
                batch_labels = labels[start:start + batch_size]

                batch_images = np.zeros((len(batch_idx), *target_size, 3), dtype='float32')
                for i, fp in enumerate(batch_files):
                    im = cv.imread(fp)
                    if im is not None:
                        im = im[:, :, ::-1]  # BGR -> RGB
                        im = cv.resize(im, target_size)
                        batch_images[i] = im / 255.0
                yield batch_images, batch_labels
    return gen


def raw_data_to_dataframe(data_path):
    """
    Build a DataFrame from a folder of plant disease images.

    Scans subdirectories (each named label_all), reads valid images,
    and returns a DataFrame with columns: label_all, label_plant,
    label_disease, filename, filepath.
    """
    print('data_path:', data_path)
    print('\n--------------------\n * RAW DATA TO DF * \n--------------------\n')

    mysubfolders = os.listdir(data_path)

    data_dico = {
        'label_all': [],
        'label_plant': [],
        'label_disease': [],
        'filename': [],
        'filepath': [],
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
                im = im[:, :, ::-1]  # BGR → RGB
                data_dico['label_all'].append(subdir)
                data_dico['label_plant'].append(subdir[0: subdir.find('___')])
                data_dico['label_disease'].append(subdir[subdir.find('___') + 3:])
                data_dico['filename'].append(file)
                data_dico['filepath'].append(file_path)

    # NA image count
    print('\n------------------------\n * data/image removed * \n------------------------\n')
    print('Amout data/image removed from the RAW dataset:', len(dataNA_dico['label_all']))
    count_NA = Counter(dataNA_dico['label_all'])
    print('\nNA files per Label_all count:\n', count_NA)

    data_df = pd.DataFrame(data_dico)
    print('\nAmout data/image read/computed with success:', len(data_df.label_all))
    print('\ndf features:', data_df.columns)

    print('\n--------------------\n * data_df issued * \n--------------------\n')
    print('does not include image but image_path!')

    return data_df
