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
    if not os.path.isfile(db_path + '.bak'):
        print('Creating backup file')
        shutil.copyfile(db_path, db_path + '.bak')

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    column_dtypes = {}
    result = print_and_exec(c, 'PRAGMA table_info(''PROTOCOL'')')
    for row in result:
        column_dtypes[row[1].lower()] = row[2]

    print('\nCreating empty PROTOCOL2 table')
    print_and_exec(c, 'DROP TABLE IF EXISTS PROTOCOL2')
    print_and_exec(c, 'CREATE TABLE PROTOCOL2 (id INTEGER PRIMARY KEY)')
    print_and_exec(c, 'INSERT INTO protocol2 (id) SELECT id FROM protocol')
    conn.commit()

    df = pd.read_csv('fields_of_interest.txt')
    query_add1 = 'ALTER TABLE protocol2 ADD COLUMN %s %s'
    query_add2 = 'ALTER TABLE protocol2 ADD COLUMN %s boolean'
    query_copy = 'UPDATE protocol2 SET %s = (SELECT p.%s FROM protocol p WHERE p.id = protocol2.id)'
    query_map = u'UPDATE protocol2 SET %s = ' \
                u'((SELECT p.%s FROM protocol p WHERE p.id = protocol2.id) IN (\'true\', \'м\'))'
    for row in df.iterrows():
        old_name = row[1]['old_name']
        new_name = row[1]['new_name']

        if new_name == '-':
            new_name = old_name
            print('\nCopying values from %s to %s' % (old_name, new_name))
            dtype = column_dtypes[old_name]
            print_and_exec(c, query_add1 % (new_name, dtype))
            print_and_exec(c, query_copy % (new_name, old_name))
        else:
            print('\nMapping values from %s to %s' % (old_name, new_name))
            print_and_exec(c, query_add2 % new_name)
            print_and_exec(c, query_map % (new_name, old_name))

        conn.commit()


def select_fields_of_interest():
    db_paths = ['../data/PTD1_BASA_CLD.GDB.sqlite', '../data/PTD2_BASA_CLD.GDB.sqlite']

    for db_path in db_paths:
        process_db_file(db_path)


if __name__ == '__main__':
    select_fields_of_interest()
