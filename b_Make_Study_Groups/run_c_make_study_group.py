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

    query = 'SELECT pngfilepath, %s FROM protocol2' % class_columns[1]
    query += ' WHERE xray_validated AND (%s OR %s) ' % (class_columns[0], class_columns[1])
    return print_and_exec(c, query)


def make_study_group():
    db_paths = ['../data/PTD1_BASA_CLD.GDB.sqlite', '../data/PTD2_BASA_CLD.GDB.sqlite']

    class_of_interest = 'class_tuberculosis'
    print('\nClass of interest: ' + class_of_interest)

    match_class = class_of_interest.replace('class_', 'match_')
    class_columns = [match_class, class_of_interest]

    paths = []
    filenames = []
    class_numbers = []
    card_numbers = []
    for db_path in db_paths:
        records = process_db_file(db_path, class_columns)

        for record in records:
            path = record[0]
            paths.append(path)
            filename = os.path.split(path)[1]
            filenames.append(filename)
            class_numbers.append(record[1])
            card_numbers.append(int(filename.split('_')[0]))

    unique_card_numbers = list(set(card_numbers))
    shuffle(unique_card_numbers)

    val_portion = 0.25
    num_val_cards = int(round(len(unique_card_numbers) * val_portion))
    val_card_numbers = set(unique_card_numbers[:num_val_cards])

    is_vals = []
    val_cases_count = 0
    train_controls = 0
    train_class = 0
    val_controls = 0
    val_class = 0
    for i, card_number in enumerate(card_numbers):
        is_val = int(card_number in val_card_numbers)
        is_vals.append(is_val)
        val_cases_count += is_val

        if not is_val:
            train_controls += class_numbers[i] == 0
            train_class += class_numbers[i] == 1
        else:
            val_controls += class_numbers[i] == 0
            val_class += class_numbers[i] == 1

    print('\nUnique patients: %i, validation patient: %i' % (len(unique_card_numbers), num_val_cards))
    print('All cases: %i, validation images: %i' % (len(paths), val_cases_count))

    print('\nControls vs. Class:')
    print('Training: %i vs. %i,    Validation: %i vs. %i' % (train_controls, train_class, val_controls, val_class))

    df = pd.DataFrame(data={'path': paths, 'filename': filenames, 'class_number': class_numbers, 'is_val': is_vals})
    df.to_csv('study_group_%s_.txt' % class_of_interest, index=False)


if __name__ == '__main__':
    make_study_group()