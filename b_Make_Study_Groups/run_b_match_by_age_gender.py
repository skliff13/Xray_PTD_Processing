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

    class_of_interest = 'class_abnormal_lungs'
    print('\nClass of interest: ' + class_of_interest)

    column_dtypes = {}
    result = print_and_exec(c, 'PRAGMA table_info(PROTOCOL2)')
    for row in result:
        column_dtypes[row[1].lower()] = row[2]

    match_column_name = 'match_' + class_of_interest[6:]
    if match_column_name not in column_dtypes:
        print('\nCreating match field')
        query = 'ALTER TABLE protocol2 ADD COLUMN %s boolean' % match_column_name
        print_and_exec(c, query)

    print_and_exec(c, 'UPDATE protocol2 SET %s = 0' % match_column_name)
    conn.commit()

    query = 'SELECT id FROM protocol2 where xray_validated AND %s' % class_of_interest
    selection = print_and_exec(c, query)

    case_ids = []
    for row in selection:
        case_ids.append(row[0])

    for i, case_id in enumerate(case_ids):
        print('Matching case %i / %i' % (i + 1, len(case_ids)))

        query = ' UPDATE protocol2 SET %s = 1 '
        query += ' WHERE id = (SELECT id FROM protocol2 WHERE class_healthy AND xray_validated AND %s = 0 '
        query += ' AND is_male = (SELECT is_male FROM protocol2 WHERE id = %i) '
        query += ' ORDER BY ABS((SELECT age FROM protocol2 WHERE id = %i) - age) '
        query += ' LIMIT 1)'
        query = query % (match_column_name, match_column_name, case_id, case_id)

        print_and_exec(c, query)
        conn.commit()


def select_fields_of_interest():
    db_paths = ['../data/PTD1_BASA_CLD.GDB.sqlite', '../data/PTD2_BASA_CLD.GDB.sqlite']

    for db_path in db_paths:
        process_db_file(db_path)


if __name__ == '__main__':
    select_fields_of_interest()
