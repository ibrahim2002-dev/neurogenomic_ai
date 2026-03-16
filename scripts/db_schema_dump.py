import sqlite3
from pathlib import Path

db = Path('data/processed/neuro_genomic.db')
with sqlite3.connect(db) as conn:
    cur = conn.cursor()
    tables = ['physio_data','genomic_data','behavioral_data','separated_components','hrv_feature_matrix']
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        n = cur.fetchone()[0]
        cur.execute(f"PRAGMA table_info({t})")
        cols = [r[1] for r in cur.fetchall()]
        print(f"{t}|rows={n}|cols={cols}")
