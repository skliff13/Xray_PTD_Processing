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

    class_of_interest = 'class_tuberculosis'

    column_dtypes = {}
    result = print_and_exec(c, 'PRAGMA table_info(''PROTOCOL2'')')
    for row in result:
        column_dtypes[row[1].lower()] = row[2]

    match_column_name = 'match_' + class_of_interest[6:]
    if match_column_name not in column_dtypes:
        print('\nCreating match field')
        query = 'ALTER TABLE protocol2 ADD COLUMN %s boolean' % match_column_name
        print_and_exec(c, query)

    print_and_exec(c, 'UPDATE protocol2 SET %s = 0' % match_column_name)
    conn.commit()

    query = 'SELECT id FROM protocol2 where %s' % class_of_interest
    selection = print_and_exec(c, query)

    ids = []
    for row in selection:
        ids.append(row[0])

    for i, id in enumerate(ids):
        print('Matching case %i / %i' % (i + 1, len(ids)))

        query = ' UPDATE protocol2 SET match_tuberculosis = 1 ' \
            ' WHERE id = (SELECT id FROM protocol2 WHERE class_healthy AND %s = 0 ' \
            ' AND is_male = (SELECT is_male FROM protocol2 WHERE id = %i) ' \
            ' ORDER BY ABS(SUBSTR((SELECT dateroshd FROM protocol2 WHERE id = %i), 7, 4) - SUBSTR(dateroshd, 7, 4)) ' \
            ' LIMIT 1)'
        query = query % (match_column_name, id, id)

        print_and_exec(c, query)
        conn.commit()


def select_fields_of_interest():
    db_paths = ['../data/PTD1_BASA_CLD.GDB.sqlite', '../data/PTD2_BASA_CLD.GDB.sqlite']

    for db_path in db_paths:
        process_db_file(db_path)


if __name__ == '__main__':
    select_fields_of_interest()
