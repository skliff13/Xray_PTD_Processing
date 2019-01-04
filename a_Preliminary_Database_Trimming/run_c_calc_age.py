# -*- coding: utf-8 -*-
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

    if 'age' not in column_dtypes:
        print('\nCreating age column')
        query = 'ALTER TABLE protocol2 ADD COLUMN age integer'
        print_and_exec(c, query)

    print('\nUpdating age values')
    query = 'UPDATE protocol2 SET age = (substr(dateissl, 7, 4) - substr(dateroshd, 7, 4))'
    print_and_exec(c, query)
    conn.commit()


def calc_age():
    db_paths = ['../data/PTD1_BASA_CLD.GDB.sqlite', '../data/PTD2_BASA_CLD.GDB.sqlite']
    db_paths = db_paths[1:]

    for db_path in db_paths:
        process_db_file(db_path)


if __name__ == '__main__':
    calc_age()
