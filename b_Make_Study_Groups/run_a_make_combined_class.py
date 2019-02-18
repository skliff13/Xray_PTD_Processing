# -*- coding: utf-8 -*-
import os
import shutil
import sqlite3
import pandas as pd


def print_and_exec(cursor, query):
    print('>> "' + query + '"')
    return cursor.execute(query)


def add_prefix(s):
    return 'class_' + s


def process_db_file(db_path):
    print('\nProcessing "%s"' % db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    column_dtypes = {}
    result = print_and_exec(c, 'PRAGMA table_info(PROTOCOL2)')
    for row in result:
        column_dtypes[row[1].lower()] = row[2]

    classes_to_combine = ['pneumosclerosis', 'emphysema', 'fibrosis', 'pneumonia', 'focal_shadows',
                          'bronchitis', 'tuberculosis']
    print('\nClasses to combine: ' + ', '.join(classes_to_combine))

    combined_column_name = 'class_abnormal_lungs'
    if combined_column_name not in column_dtypes:
        print('\nCreating match field')
        query = 'ALTER TABLE protocol2 ADD COLUMN %s boolean' % combined_column_name
        print_and_exec(c, query)

    condition = ' OR '.join(list(map(add_prefix, classes_to_combine)))
    query = 'UPDATE protocol2 SET %s = (%s)' % (combined_column_name, condition)
    print_and_exec(c, query)
    conn.commit()


def select_fields_of_interest():
    db_paths = ['../data/PTD1_BASA_CLD.GDB.sqlite', '../data/PTD2_BASA_CLD.GDB.sqlite']

    for db_path in db_paths:
        process_db_file(db_path)


if __name__ == '__main__':
    select_fields_of_interest()
