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

    if 'class_healthy' not in column_dtypes:
        print('\nCreating class_healthy data column')
        query = 'ALTER TABLE protocol2 ADD COLUMN class_healthy boolean'
        print_and_exec(c, query)

    print('\nUpdating class_healthy values')
    query = 'UPDATE protocol2 SET class_healthy = (opisanie in ('
    query += '\'Лёгочные поля чистые. Корни структурны.\', '
    query += '\'Очаговые и инфильтративные тени не определяются. Корни легких структурные.\', '
    query += '\'При компьютерной рентгенографии патологических изменений не обнаружено.\'))'
    print_and_exec(c, query)
    conn.commit()

    df = pd.read_csv('fields_of_interest.txt')
    print('\nEnsuring healthy cases')
    for row in df.iterrows():
        new_name = row[1]['new_name']

        if new_name.startswith('class_'):
            query = 'UPDATE protocol2 SET class_healthy = 0 WHERE %s = 1' % new_name
            print_and_exec(c, query)

    conn.commit()


def mark_healthy():
    db_paths = ['../data/PTD1_BASA_CLD.GDB.sqlite', '../data/PTD2_BASA_CLD.GDB.sqlite']

    for db_path in db_paths:
        process_db_file(db_path)


if __name__ == '__main__':
    mark_healthy()
