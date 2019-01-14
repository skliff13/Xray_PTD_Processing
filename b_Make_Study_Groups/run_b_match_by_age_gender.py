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

    query = 'SELECT age, is_male, count(1) FROM protocol2 where xray_validated AND %s GROUP BY age, is_male'
    query = query % class_of_interest
    selection = print_and_exec(c, query)

    age_gender_counts = []
    for row in selection:
        age_gender_counts.append([row[0], row[1], row[2]])

    for i, age_gender_count in enumerate(age_gender_counts):
        print('Matching age-gender %i / %i' % (i + 1, len(age_gender_counts)))

        age = age_gender_count[0]
        is_male = age_gender_count[1]
        count = age_gender_count[2]

        age_span, validated_xrays_only = get_minimum_selection_parameters(age, c, count, is_male, match_column_name)

        query = ' UPDATE protocol2 SET %s = 1 WHERE id IN ' % match_column_name
        query += ' (SELECT id FROM protocol2 WHERE class_healthy AND %s = 0 ' % match_column_name
        if validated_xrays_only:
            query += ' AND xray_validated '
        else:
            query += ' AND (xray_validated OR xray_validated IS NULL) '
        query += ' AND ABS(age - %i) <= %i AND is_male = %i ' % (age, age_span, is_male)
        if age_span > 0:
            query += ' ORDER BY ABS(age - %i) ' % age
        query += ' LIMIT %i)' % count

        print_and_exec(c, query)
        conn.commit()


def get_minimum_selection_parameters(age, c, count, is_male, match_column_name):
    age_span = 0
    validated_xrays_only = True

    while True:
        query = 'SELECT COUNT(1) FROM protocol2 WHERE class_healthy AND %s = 0 ' % match_column_name
        if validated_xrays_only:
            query += ' AND xray_validated '
        else:
            query += ' AND (xray_validated OR xray_validated IS NULL) '
        query += ' AND ABS(age - %i) <= %i AND is_male = %i ' % (age, age_span, is_male)
        result = c.execute(query)

        for row in result:
            if row[0] >= count:
                return age_span, validated_xrays_only

        if validated_xrays_only:
            validated_xrays_only = False
        else:
            age_span += 1


def select_fields_of_interest():
    db_paths = ['../data/PTD1_BASA_CLD.GDB.sqlite', '../data/PTD2_BASA_CLD.GDB.sqlite']

    for db_path in db_paths:
        process_db_file(db_path)


if __name__ == '__main__':
    select_fields_of_interest()
