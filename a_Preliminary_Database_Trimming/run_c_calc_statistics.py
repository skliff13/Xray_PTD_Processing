import sqlite3
import pandas as pd


def print_and_exec(cursor, query):
    print('>> "' + query + '"')
    return cursor.execute(query)


def print_matrix(class_fields, matrix):

    print('\nMatrix of class co-occurrence:\n')

    s = '%15s ' % '~'
    for class_field in class_fields:
        s += '%15s ' % class_field[6:]
    print(s)
    for class_field1 in class_fields:
        s = '%15s ' % class_field1[6:]
        matrix_row = matrix[class_fields.index(class_field1)]
        for class_field2 in class_fields:
            s += '%15s ' % str(matrix_row[class_fields.index(class_field2)])

        print(s)


def process_db_file(db_path):
    print('\n\Processing "%s"' % db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    df = pd.read_csv('fields_of_interest_trial.txt')
    class_fields = ['class_healthy']
    for row in df.iterrows():
        new_name = row[1]['new_name']

        if new_name.startswith('class_'):
            class_fields.append(new_name)
    matrix = []
    for class_field1 in class_fields:
        matrix_row = []
        for class_field2 in class_fields:
            query = 'SELECT COUNT(1) FROM protocol2 WHERE %s AND %s' % (class_field1, class_field2)
            result = print_and_exec(c, query)
            for row in result:
                matrix_row.append(row[0])

        matrix.append(matrix_row)
    print_matrix(class_fields, matrix)


def calc_statistics():
    # db_paths = ['../data/PTD1_BASA_CLD.GDB.sqlite', '../data/PTD2_BASA_CLD.GDB.sqlite']
    db_paths = ['../data/PTD2_BASA_CLD.GDB.sqlite']

    for db_path in db_paths:
        process_db_file(db_path)


if __name__ == '__main__':
    calc_statistics()
