# -*- coding: utf-8 -*-
import os
import shutil
import sqlite3
import pandas as pd


def print_and_exec(cursor, query):
    print('>> "' + query + '"')
    return cursor.execute(query)


def process_db_file(db_path):
    print('\nProcessing "%s"' % db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    column_dtypes = {}
    result = print_and_exec(c, 'PRAGMA table_info(PROTOCOL2)')
    for row in result:
        column_dtypes[row[1].lower()] = row[2]

    validation_column_name = 'xray_validated'
    if validation_column_name not in column_dtypes:
        print('\nCreating X-ray validation field')
        query = 'ALTER TABLE protocol2 ADD COLUMN %s boolean' % validation_column_name
        print_and_exec(c, query)
        conn.commit()

    _, ptd = os.path.split(db_path)
    ptd = ptd[:4]

    batch_size = 10000
    df = pd.read_csv('../data/files_info_all_together.txt')
    counter = 0
    values = []
    batch_counter = 0
    for row in df.iterrows():
        path = row[1]['partDir'] + '\\' + row[1]['gdbDir'] + '\\' + row[1]['originalFilename']
        if path.startswith(ptd):
            validated = 1 - row[1]['isDeleted']
            values.append('(\'' + path + '\', ' + str(validated) + ')')
            counter += 1

        if counter == batch_size:
            batch_counter += 1
            update_validation_batch(c, values)
            print('Batch %i / %i' % (batch_counter, df.shape[0] // batch_size))
            values = []
            counter = 0

    if values:
        update_validation_batch(c, values)
        print('Final batch')

    conn.commit()


def update_validation_batch(c, values):
    values_str = ', \n'.join(values)
    query = 'WITH Tmp(path, valid) as (VALUES\n%s) \n' % values_str
    query += 'UPDATE protocol2 SET xray_validated = (SELECT valid FROM Tmp WHERE protocol2.pngfilepath = Tmp.path) \n'
    query += 'WHERE pngfilepath IN (SELECT path from Tmp)'
    c.execute(query)


def add_validation_flag():
    db_paths = ['../data/PTD1_BASA_CLD.GDB.sqlite', '../data/PTD2_BASA_CLD.GDB.sqlite']

    for db_path in db_paths:
        process_db_file(db_path)


if __name__ == '__main__':
    add_validation_flag()
