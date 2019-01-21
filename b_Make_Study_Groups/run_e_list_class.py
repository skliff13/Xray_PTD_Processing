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


def process_db_file(db_path, class_of_interest):
    print('\nProcessing "%s"' % db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    query = 'SELECT pngfilepath, age, is_male FROM protocol2'
    query += ' WHERE (xray_validated OR xray_validated IS NULL)'
    query += ' AND %s ' % class_of_interest
    return print_and_exec(c, query)


def list_class():
    db_paths = ['../data/PTD1_BASA_CLD.GDB.sqlite', '../data/PTD2_BASA_CLD.GDB.sqlite']

    class_of_interest = 'class_healthy'
    print('\nClass of interest: ' + class_of_interest)

    met_card_numbers = set()
    paths = []
    filenames = []
    card_numbers = []
    ages = []
    is_males = []
    for db_path in db_paths:
        records = process_db_file(db_path, class_of_interest)

        for record in records:
            path = record[0]
            filename = os.path.split(path)[1]
            card_number = int(filename.split('_')[0])

            if card_number not in met_card_numbers:
                paths.append(path)
                filenames.append(filename)
                card_numbers.append(card_number)
                ages.append(record[1])
                is_males.append(record[2])

                met_card_numbers.add(card_number)

    zeros = [0] * len(paths)
    df = pd.DataFrame(data={'path': paths, 'filename': filenames, 'age': ages, 'is_male': is_males,
                            'class_number': zeros, 'is_val': zeros})
    out_filepath = '../data/list_%s.txt' % class_of_interest
    print('Saving data to "%s"' % out_filepath)
    df.to_csv(out_filepath, index=False)


if __name__ == '__main__':
    list_class()
