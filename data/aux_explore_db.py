import sqlite3

db_path = 'data/PTD2_BASA_CLD.GDB.sqlite'

conn = sqlite3.connect(db_path)
c = conn.cursor()

db_columns = []
result = c.execute('PRAGMA table_info(''PROTOCOL'')')
for row in result:
    db_columns.append(row[1])

print(tuple(db_columns))

query = 'SELECT * FROM protocol WHERE id < 100'
for row in c.execute(query):
    print(row)
