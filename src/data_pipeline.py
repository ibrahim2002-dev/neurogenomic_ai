# Data Pipeline — loads physiological, genomic, and behavioral data

import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3

try:
    import wfdb
except ImportError:
    wfdb = None


class DataPipeline:

    def __init__(self, data_dir="../data", db_path="neuro_genomic.db"):
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path)
        self.physio_data = None
        self.genomic_data = None
        self.behavioral_data = None

    # Resolve DB path with backward compatibility for old data/processed paths
    def _resolve_db_path(self, db_path=None):
        candidate = Path(db_path) if db_path is not None else self.db_path
        if candidate.is_absolute():
            return candidate

        # Prefer direct relative path first (project/workdir), then fallback to data_dir
        if candidate.exists():
            return candidate

        data_candidate = self.data_dir / candidate
        if data_candidate.exists():
            return data_candidate

        return candidate

    # Load CSV from data/physio/
    def load_physiological_data(self, filename):
        path = self.data_dir / "physio" / filename
        self.physio_data = pd.read_csv(path)
        return self.physio_data

    # Load CSV from data/genomic/
    def load_genomic_data(self, filename):
        path = self.data_dir / "genomic" / filename
        self.genomic_data = pd.read_csv(path)
        return self.genomic_data

    # Load CSV from data/behavioral/
    def load_behavioral_data(self, filename):
        path = self.data_dir / "behavioral" / filename
        self.behavioral_data = pd.read_csv(path)
        return self.behavioral_data

    # Return shape + column names for every loaded dataset
    def get_dataset_summary(self):
        summary = {}
        for name, df in [('physiological', self.physio_data),
                         ('genomic', self.genomic_data),
                         ('behavioral', self.behavioral_data)]:
            if df is not None:
                summary[name] = {'shape': df.shape, 'columns': list(df.columns)}
        return summary

    # Load all required local CSV files in one call (no downloads)
    def load_required_local_data(
        self,
        physio_file='sample_physio.csv',
        genomic_file='sample_genomic.csv',
        behavioral_file='sample_behavioral.csv'
    ):
        self.load_physiological_data(physio_file)
        self.load_genomic_data(genomic_file)
        self.load_behavioral_data(behavioral_file)
        return {
            'physio': self.physio_data,
            'genomic': self.genomic_data,
            'behavioral': self.behavioral_data,
        }

    # Create/refresh a local SQLite database from local CSV files
    def bootstrap_local_database(
        self,
        db_path=None,
        physio_file='sample_physio.csv',
        genomic_file='sample_genomic.csv',
        behavioral_file='sample_behavioral.csv'
    ):
        db_file = self._resolve_db_path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        physio_df = pd.read_csv(self.data_dir / 'physio' / physio_file)
        genomic_df = pd.read_csv(self.data_dir / 'genomic' / genomic_file)
        behavioral_df = pd.read_csv(self.data_dir / 'behavioral' / behavioral_file)

        # Clean object columns (e.g., accidental extra spaces in CSV fields)
        for df in (physio_df, genomic_df, behavioral_df):
            obj_cols = df.select_dtypes(include='object').columns
            for col in obj_cols:
                df[col] = df[col].astype(str).str.strip()

        with sqlite3.connect(db_file) as conn:
            physio_df.to_sql('physio_data', conn, if_exists='replace', index=False)
            genomic_df.to_sql('genomic_data', conn, if_exists='replace', index=False)
            behavioral_df.to_sql('behavioral_data', conn, if_exists='replace', index=False)

        return db_file

    # Load one table directly from SQLite database file
    def load_table_from_database(self, table_name, db_path=None):
        db_file = self._resolve_db_path(db_path)
        with sqlite3.connect(db_file) as conn:
            return pd.read_sql_query(f'SELECT * FROM {table_name}', conn)

    # Load custom SQL query directly from SQLite database file
    def load_query_from_database(self, query, db_path=None):
        db_file = self._resolve_db_path(db_path)
        with sqlite3.connect(db_file) as conn:
            return pd.read_sql_query(query, conn)

    # Load all required datasets directly from database tables
    def load_required_database_data(
        self,
        db_path=None,
        physio_table='physio_data',
        genomic_table='genomic_data',
        behavioral_table='behavioral_data'
    ):
        self.physio_data = self.load_table_from_database(physio_table, db_path=db_path)
        self.genomic_data = self.load_table_from_database(genomic_table, db_path=db_path)
        self.behavioral_data = self.load_table_from_database(behavioral_table, db_path=db_path)

        return {
            'physio': self.physio_data,
            'genomic': self.genomic_data,
            'behavioral': self.behavioral_data,
        }

    # Store any DataFrame in SQLite for reproducible downstream use
    def store_dataframe_in_database(self, df, table_name, db_path=None, if_exists='replace'):
        db_file = self._resolve_db_path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_file) as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        return db_file

    # Download one PhysioNet record using wfdb and return a tidy DataFrame
    def download_physionet_record(self, database, record_name, channels=None):
        if wfdb is None:
            raise ImportError(
                'wfdb is not installed. Install dependencies from requirements.txt first.'
            )

        record = wfdb.rdrecord(record_name, pn_dir=database, channels=channels)
        signal_data = record.p_signal
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(-1, 1)

        sig_names = record.sig_name or [f'ch_{i}' for i in range(signal_data.shape[1])]
        df = pd.DataFrame(signal_data, columns=sig_names)
        df.insert(0, 'sample_index', np.arange(len(df), dtype=int))
        df.insert(1, 'time_sec', df['sample_index'] / float(record.fs))
        df['record_name'] = record_name
        df['database'] = database
        df['sampling_rate'] = float(record.fs)
        return df

    # Download PhysioNet record and persist it directly to SQLite
    def ingest_physionet_record_to_database(
        self,
        database,
        record_name,
        table_name=None,
        channels=None,
        db_path=None,
        if_exists='replace'
    ):
        physio_df = self.download_physionet_record(
            database=database,
            record_name=record_name,
            channels=channels,
        )
        if table_name is None:
            safe_record = record_name.replace('/', '_').replace('-', '_')
            table_name = f'physionet_{database}_{safe_record}'

        db_file = self.store_dataframe_in_database(
            physio_df,
            table_name=table_name,
            db_path=db_path,
            if_exists=if_exists,
        )
        return {
            'table_name': table_name,
            'db_path': db_file,
            'rows': len(physio_df),
            'columns': list(physio_df.columns),
        }

    # Load ECG channel from a PhysioNet-ingested SQLite table
    def load_physionet_channel(self, table_name, signal_column, db_path=None):
        query = (
            f'SELECT sample_index, time_sec, {signal_column} '
            f'FROM {table_name} ORDER BY sample_index'
        )
        return self.load_query_from_database(query, db_path=db_path)
