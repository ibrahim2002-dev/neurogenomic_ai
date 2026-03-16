import sqlite3
from pathlib import Path

DB_PATH = Path('data/processed/neuro_genomic.db')
print(f'DB exists: {DB_PATH.exists()} at {DB_PATH.resolve()}')

if not DB_PATH.exists():
    raise SystemExit(1)

with sqlite3.connect(DB_PATH) as conn:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    print('Tables:', tables)

    for t in ['physio_data', 'genomic_data', 'behavioral_data', 'separated_components', 'hrv_feature_matrix']:
        if t in tables:
            cur.execute(f'SELECT COUNT(*) FROM {t}')
            print(f'{t} rows: {cur.fetchone()[0]}')
        else:
            print(f'{t} rows: table not found')
