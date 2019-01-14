# -*- coding: utf-8 -*-
import os
import shutil
import sqlite3
import numpy as np
import pandas as pd


def print_and_exec(cursor, query):
    print('>> "' + query + '"')
    return cursor.execute(query)


def process_db_file(db_path, class_columns):
    print('\nProcessing "%s"' % db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    all_ages = [[[], []], [[], []]]
    for is_male in range(2):
        for ic, class_column in enumerate(class_columns):
            query = ' SELECT age FROM protocol2'
            query += ' WHERE is_male = %i' % is_male
            query += ' AND %s' % class_column
            query += ' AND (xray_validated OR xray_validated IS NULL) AND age >= 10 AND age <= 69'
            result = print_and_exec(c, query)

            ages = []
            for row in result:
                ages.append(row[0])

            all_ages[is_male][ic] = ages

    return all_ages


def select_fields_of_interest():
    db_paths = ['../data/PTD1_BASA_CLD.GDB.sqlite', '../data/PTD2_BASA_CLD.GDB.sqlite']

    class_of_interest = 'class_abnormal_lungs'
    print('\nClass of interest: ' + class_of_interest)

    match_class = class_of_interest.replace('class_', 'match_')
    class_columns = [match_class, class_of_interest]

    all_ages = [[[], []], [[], []]]
    for db_path in db_paths:
        ages = process_db_file(db_path, class_columns)

        for is_male in range(2):
            for ic in range(2):
                all_ages[is_male][ic] += ages[is_male][ic]

    for is_male in range(2):
        for ic in range(2):
            ages = all_ages[is_male][ic]
            ages = np.array(ages)

            s = 'is_male=%i, %s: NUM=%i, MEAN=%.2f, STD=%.2f'
            s = s % (is_male, class_columns[ic], ages.shape[0], float(np.mean(ages)), float(np.std(ages)))
            print(s)


if __name__ == '__main__':
    select_fields_of_interest()
