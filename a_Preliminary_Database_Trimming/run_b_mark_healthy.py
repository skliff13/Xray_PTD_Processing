import sqlite3
import pandas as pd


def print_and_exec(cursor, query):
    print('>> "' + query + '"')
    cursor.execute(query)


def main():
    db_path = '../data/PTD2_BASA_CLD.GDB.sqlite'

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    column_dtypes = {}
    result = c.execute('PRAGMA table_info(''PROTOCOL2'')')
    for row in result:
        column_dtypes[row[1].lower()] = row[2]

    if 'class_healthy' not in column_dtypes:
        print('\nCreating class_healthy data column')
        query = 'ALTER TABLE protocol2 ADD COLUMN class_healthy boolean'
        print_and_exec(c, query)

    print('\nUpdating class_healthy values')
    query = 'UPDATE protocol2 SET class_healthy = (opisanie in (\'Лёгочные поля чистые. Корни структурны.\'))'
    print_and_exec(c, query)
    conn.commit()

    df = pd.read_csv('fields_of_interest_trial.txt')
    print('\nEnsuring healthy cases')
    for row in df.iterrows():
        new_name = row[1]['new_name']

        if new_name.startswith('class_'):
            query = 'UPDATE protocol2 SET class_healthy = 0 WHERE %s = 1' % new_name
            print_and_exec(c, query)

    conn.commit()


if __name__ == '__main__':
    main()
