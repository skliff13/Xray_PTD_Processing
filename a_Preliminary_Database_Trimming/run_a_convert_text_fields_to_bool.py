import sqlite3
import pandas as pd


def print_and_exec(c, query):
    print(query)
    c.execute(query)


db_path = '../data/PTD2_BASA_CLD.GDB.sqlite'

conn = sqlite3.connect(db_path)
c = conn.cursor()

column_dtypes = {}
result = c.execute('PRAGMA table_info(''PROTOCOL'')')
for row in result:
    column_dtypes[row[1].lower()] = row[2]

df = pd.read_csv('fields_of_interest.txt')

print('# Creating empty PROTOCOL2 table')
print_and_exec(c, 'DROP TABLE IF EXISTS PROTOCOL2')
print_and_exec(c, 'CREATE TABLE PROTOCOL2 (id INTEGER PRIMARY KEY)')
print_and_exec(c, 'INSERT INTO protocol2 (id) SELECT id FROM protocol')
conn.commit()

query_add1 = 'ALTER TABLE protocol2 ADD COLUMN %s %s'
query_add2 = 'ALTER TABLE protocol2 ADD COLUMN %s boolean'
query_copy = u'UPDATE protocol2 SET %s = (SELECT %s FROM protocol)'
query_map = u'UPDATE protocol2 SET %s = (SELECT %s FROM protocol in (\'true\', \'м\'))'
for row in df.iterrows():
    old_name = row[1]['old_name']
    new_name = row[1]['new_name']

    if new_name == '-':
        new_name = old_name
        print('\n# Copying values from %s to %s' % (old_name, new_name))
        dtype = column_dtypes[old_name]
        print_and_exec(c, query_add1 % (new_name, dtype))
        print_and_exec(c, query_copy % (new_name, old_name))
    else:
        print('\n# Mapping values from %s to %s' % (old_name, new_name))
        print_and_exec(c, query_add2 % new_name)
        print_and_exec(c, query_copy % (new_name, old_name))

    conn.commit()




