# -*- coding: utf-8 -*-
import os
import shutil
import sqlite3
import numpy as np
import pandas as pd
from random import shuffle


def print_and_exec(cursor, query):
    print('>> "' + query + '"')
    return cursor.execute(query)


def process_db_file(db_path, class_columns):
    print('\nProcessing "%s"' % db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    query = 'SELECT pngfilepath, %s, age, is_male, class_healthy, ' % class_columns[1]
    query += ' class_pneumosclerosis, class_emphysema, class_fibrosis, class_pneumonia, '
    query += ' class_focal_shadows, class_bronchitis, class_tuberculosis '
    query += ' FROM protocol2 '
    query += ' WHERE (xray_validated OR xray_validated IS NULL) AND age >= 10 AND age <= 69'
    query += ' AND (%s OR %s) ' % (class_columns[0], class_columns[1])
    return print_and_exec(c, query)


def make_study_group():
    db_paths = ['../data/PTD1_BASA_CLD.GDB.sqlite', '../data/PTD2_BASA_CLD.GDB.sqlite']

    # class_of_interest = 'class_tuberculosis'
    class_of_interest = 'class_abnormal_lungs'
    print('\nClass of interest: ' + class_of_interest)

    match_class = class_of_interest.replace('class_', 'match_')
    class_columns = [match_class, class_of_interest]

    paths = []
    filenames = []
    class_numbers = []
    card_numbers = []
    ages = []
    is_males = []
    healthy = []
    pneumosclerosis = []
    emphysema = []
    fibrosis = []
    pneumonia = []
    focal_shadows = []
    bronchitis = []
    tuberculosis = []

    for db_path in db_paths:
        records = process_db_file(db_path, class_columns)

        for record in records:
            path = record[0]
            paths.append(path)
            filename = os.path.split(path)[1]
            filenames.append(filename)
            class_numbers.append(record[1])
            card_numbers.append(int(filename.split('_')[0]))
            ages.append(record[2])
            is_males.append(record[3])

            healthy.append(record[4])
            pneumosclerosis.append(record[5])
            emphysema.append(record[6])
            fibrosis.append(record[7])
            pneumonia.append(record[8])
            focal_shadows.append(record[9])
            bronchitis.append(record[10])
            tuberculosis.append(record[11])

    unique_card_numbers = list(set(card_numbers))
    shuffle(unique_card_numbers)

    val_portion = 0.25
    num_val_cards = int(round(len(unique_card_numbers) * val_portion))
    val_card_numbers = set(unique_card_numbers[:num_val_cards])

    is_vals = []
    val_cases_count = 0
    train_controls_count = 0
    train_class_count = 0
    val_controls_count = 0
    val_class_count = 0
    for i, card_number in enumerate(card_numbers):
        is_val = int(card_number in val_card_numbers)
        is_vals.append(is_val)
        val_cases_count += is_val

        if not is_val:
            train_controls_count += class_numbers[i] == 0
            train_class_count += class_numbers[i] == 1
        else:
            val_controls_count += class_numbers[i] == 0
            val_class_count += class_numbers[i] == 1

    print('\nUnique patients: %i, validation patient: %i' % (len(unique_card_numbers), num_val_cards))
    print('All cases: %i, validation images: %i' % (len(paths), val_cases_count))

    print('\nControls vs. Class:')
    print('Training: %i vs. %i,    Validation: %i vs. %i' % (train_controls_count, train_class_count,
                                                             val_controls_count, val_class_count))

    df = pd.DataFrame(data={'path': paths, 'filename': filenames, 'age': ages, 'is_male': is_males,
                            'class_number': class_numbers, 'is_val': is_vals,'healthy': healthy,
                            'pneumosclerosis': pneumosclerosis, 'emphysema': emphysema, 'fibrosis': fibrosis,
                            'pneumonia': pneumonia, 'focal_shadows': focal_shadows,
                            'bronchitis': bronchitis, 'tuberculosis': tuberculosis})
    out_filepath = '../data/study_group_%s_.txt' % class_of_interest
    print('Saving data to "%s"' % out_filepath)
    df.to_csv(out_filepath, index=False)


if __name__ == '__main__':
    make_study_group()
