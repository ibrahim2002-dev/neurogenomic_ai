import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

try:
    import wfdb
except ImportError:
    wfdb = None

from data_pipeline import DataPipeline
from feature_extraction import HRVExtractor
from model import CognitiveStateClassifier
from preprocessing import ECGPreprocessor
from signal_separation import ComponentAnalyzer, SignalSeparator


DATABASE_ALIASES = {
    'adecg': 'adfecgdb',
    'longecgdb': 'ltdb',
    'nifecgdb': 'nifecgdb',
}


def _resolve_database_name(name: str) -> str:
    return DATABASE_ALIASES.get(name.lower().strip(), name.lower().strip())


def _resolve_record_name(database: str, explicit_record: str | None) -> str:
    if explicit_record:
        return explicit_record
    if wfdb is None:
        return '100'
    try:
        records = wfdb.get_record_list(database)
        if records:
            return records[0]
    except Exception:
        pass
    return '100'


def _pick_best_numeric_column(df: pd.DataFrame, excluded: set[str]) -> str:
    candidates = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if not candidates:
        raise ValueError('No numeric ECG signal column found in ingested PhysioNet table.')
    return max(candidates, key=lambda c: int(df[c].notna().sum()))


def _auto_pick_signal_column(df: pd.DataFrame) -> str:
    excluded = {
        'sample_index',
        'time_sec',
        'record_name',
        'database',
        'sampling_rate',
        'source_database',
        'source_record',
    }
    return _pick_best_numeric_column(df, excluded)


def _numeric_signal_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        'sample_index',
        'time_sec',
        'record_name',
        'database',
        'sampling_rate',
        'source_database',
        'source_record',
    }
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def _build_target_labels(features: pd.DataFrame) -> pd.Series:
    score = (
        0.35 * features['fet_heart_rate_mean']
        - 0.15 * features['mat_heart_rate_mean']
        + 0.30 * features['fet_rmssd']
        - 0.10 * features['mat_rmssd']
        + 0.25 * features['fet_pnn50']
        - 0.05 * features['mat_pnn50']
    )
    q1, q2 = np.quantile(score, [0.33, 0.66])
    labels = np.where(score <= q1, 'low_maturity', np.where(score <= q2, 'mid_maturity', 'high_maturity'))
    return pd.Series(labels, name='target')


def _build_default_genomic_table(n_rows: int) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame(
        {
            'sample_id': np.arange(n_rows, dtype=int),
            'gene_expression_a': np.random.normal(0.6, 0.15, n_rows),
            'gene_expression_b': np.random.normal(0.4, 0.12, n_rows),
            'gene_expression_c': np.random.normal(0.5, 0.10, n_rows),
            'variant_load': np.random.poisson(2.0, n_rows),
        }
    )


def _build_default_behavioral_table(n_rows: int) -> pd.DataFrame:
    np.random.seed(43)
    return pd.DataFrame(
        {
            'task_id': np.arange(n_rows, dtype=int),
            'click_rate': np.random.normal(4.0, 1.0, n_rows),
            'typing_speed': np.random.normal(35.0, 7.0, n_rows),
            'error_rate': np.clip(np.random.normal(0.12, 0.05, n_rows), 0.0, 1.0),
            'focus_score': np.clip(np.random.normal(0.65, 0.15, n_rows), 0.0, 1.0),
        }
    )


def main():
    parser = argparse.ArgumentParser(description='Download PhysioNet ECG, process, separate signals, extract HRV features, and train NN model.')
    parser.add_argument(
        '--databases',
        default='nifecgdb,longecgdb,adecg',
        help='Comma-separated PhysioNet dataset aliases/names to ingest (default: nifecgdb,longecgdb,adecg).',
    )
    parser.add_argument(
        '--database',
        default=None,
        help='Optional single dataset name; if provided, overrides --databases.',
    )
    parser.add_argument(
        '--record',
        default=None,
        help='Optional record identifier applied to all selected databases. If omitted, first available record is used per database.',
    )
    parser.add_argument('--signal-column', default=None, help='Specific ECG column to use after ingestion.')
    parser.add_argument('--window-sec', type=int, default=10, help='Window length in seconds for HRV extraction.')
    parser.add_argument('--genomic-csv', default=None, help='Optional genomic CSV path; overrides GEO download.')
    parser.add_argument('--genomic-db', default='GSE55750', help='NCBI GEO accession for genomic data (default: GSE55750). See https://www.ncbi.nlm.nih.gov/geo/')
    parser.add_argument('--behavioral-csv', default=None, help='Optional behavioral CSV path; overrides PhysioNet CLAS download.')
    parser.add_argument('--behavioral-db', default='clas', help='PhysioNet behavioral database (default: clas). See https://physionet.org/content/clas/1.0.0/')
    parser.add_argument('--behavioral-record', default='001', help='Record identifier within behavioral-db (default: 001).')
    parser.add_argument('--no-real-data', action='store_true', help='Skip external downloads and use synthetic genomic/behavioral data.')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    db_path = root / 'data' / 'processed' / 'neuro_genomic.db'
    artifacts_dir = root / 'results' / 'models'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DataPipeline(data_dir=root / 'data', db_path=db_path)

    if args.database:
        requested_databases = [args.database]
    else:
        requested_databases = [d.strip() for d in args.databases.split(',') if d.strip()]
    if not requested_databases:
        raise ValueError('No databases were provided. Pass --database or --databases.')

    raw_parts = []
    source_rows = []
    for db_name in requested_databases:
        resolved_db = _resolve_database_name(db_name)
        record_name = _resolve_record_name(resolved_db, args.record)
        table_name = f'physio_{resolved_db}_{str(record_name).replace("/", "_").replace("-", "_")}'

        pipeline.ingest_physionet_record_to_database(
            database=resolved_db,
            record_name=record_name,
            table_name=table_name,
            db_path=db_path,
            if_exists='replace',
        )
        part = pipeline.load_table_from_database(table_name, db_path=db_path)
        part['source_database'] = resolved_db
        part['source_record'] = str(record_name)
        raw_parts.append(part)
        source_rows.append({'database': resolved_db, 'record': str(record_name), 'rows': int(len(part))})

    raw_df = pd.concat(raw_parts, ignore_index=True, sort=False)
    pipeline.store_dataframe_in_database(raw_df, 'physio_data', db_path=db_path, if_exists='replace')

    # --- Genomic data: NCBI GEO (default GSE55750) ---
    # Dataset URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE55750
    if args.genomic_csv:
        genomic_df = pd.read_csv(args.genomic_csv)
        pipeline.store_dataframe_in_database(genomic_df, 'genomic_data', db_path=db_path, if_exists='replace')
    elif not args.no_real_data:
        print(f'Downloading genomic data from NCBI GEO ({args.genomic_db}) ...')
        print(f'  Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={args.genomic_db}')
        try:
            genomic_df = pipeline.download_genomic_geo(
                geo_accession=args.genomic_db,
                db_path=db_path,
                table_name='genomic_data',
                if_exists='replace',
            )
            print(f'  Downloaded genomic rows: {len(genomic_df)}, genes: {len([c for c in genomic_df.columns if c.startswith("gene_")])}')
        except Exception as exc:
            print(f'  Warning: GEO download failed ({exc}). Using synthetic genomic data.')
            genomic_df = _build_default_genomic_table(max(20, len(raw_df) // 1000))
            pipeline.store_dataframe_in_database(genomic_df, 'genomic_data', db_path=db_path, if_exists='replace')
    else:
        genomic_df = _build_default_genomic_table(max(20, len(raw_df) // 1000))
        pipeline.store_dataframe_in_database(genomic_df, 'genomic_data', db_path=db_path, if_exists='replace')

    # --- Behavioral data: PhysioNet CLAS (Cognitive Load, Affect and Stress) ---
    # Dataset URL: https://physionet.org/content/clas/1.0.0/
    if args.behavioral_csv:
        behavioral_df = pd.read_csv(args.behavioral_csv)
        pipeline.store_dataframe_in_database(behavioral_df, 'behavioral_data', db_path=db_path, if_exists='replace')
    elif not args.no_real_data:
        print(f'Downloading behavioral data from PhysioNet {args.behavioral_db}/{args.behavioral_record} ...')
        print(f'  Source: https://physionet.org/content/clas/1.0.0/')
        try:
            behavioral_df = pipeline.download_behavioral_physionet(
                database=args.behavioral_db,
                record_name=args.behavioral_record,
                db_path=db_path,
                table_name='behavioral_data',
                if_exists='replace',
            )
            print(f'  Downloaded behavioral rows: {len(behavioral_df)}')
        except Exception as exc:
            print(f'  Warning: CLAS download failed ({exc}). Using synthetic behavioral data.')
            behavioral_df = _build_default_behavioral_table(max(20, len(raw_df) // 1000))
            pipeline.store_dataframe_in_database(behavioral_df, 'behavioral_data', db_path=db_path, if_exists='replace')
    else:
        behavioral_df = _build_default_behavioral_table(max(20, len(raw_df) // 1000))
        pipeline.store_dataframe_in_database(behavioral_df, 'behavioral_data', db_path=db_path, if_exists='replace')

    numeric_cols = _numeric_signal_columns(raw_df)
    signal_column = args.signal_column or _auto_pick_signal_column(raw_df)
    if signal_column not in numeric_cols:
        raise ValueError(f'Signal column {signal_column} is not present in downloaded PhysioNet record.')

    secondary_candidates = [c for c in numeric_cols if c != signal_column]
    secondary_column = None
    if secondary_candidates:
        secondary_column = max(secondary_candidates, key=lambda c: int(raw_df[c].notna().sum()))

    fs = int(float(raw_df['sampling_rate'].iloc[0])) if 'sampling_rate' in raw_df.columns else 360

    primary_signal = pd.to_numeric(raw_df[signal_column], errors='coerce').interpolate(limit_direction='both')
    if primary_signal.isna().all():
        raise ValueError(f'No usable samples found in selected primary signal column: {signal_column}')
    primary_signal = primary_signal.fillna(method='ffill').fillna(method='bfill').to_numpy(dtype=float)

    if secondary_column is not None:
        secondary_signal = pd.to_numeric(raw_df[secondary_column], errors='coerce').interpolate(limit_direction='both')
        secondary_signal = secondary_signal.fillna(method='ffill').fillna(method='bfill').to_numpy(dtype=float)
    else:
        # Create a second mixed channel when only one channel is available
        secondary_signal = np.roll(primary_signal, 1) * 0.95

    mixed_signals = np.column_stack([primary_signal, secondary_signal])

    preprocessor = ECGPreprocessor(sampling_rate=fs)
    filtered = preprocessor.filter_signal(mixed_signals)
    cleaned = np.column_stack([
        preprocessor.remove_baseline_wander(filtered[:, i])
        for i in range(filtered.shape[1])
    ])
    normalized = preprocessor.normalize_signal(cleaned, method='zscore')

    separator = SignalSeparator(n_components=2, random_state=42)
    separated_components = separator.fit_transform(normalized)
    comp_info = ComponentAnalyzer.classify_components(separated_components, sampling_rate=fs)

    comp_freq = comp_info['frequencies']
    maternal_idx = min(comp_freq, key=comp_freq.get)
    fetal_idx = max(comp_freq, key=comp_freq.get)

    maternal_signal = separated_components[:, maternal_idx]
    fetal_signal = separated_components[:, fetal_idx]

    sep_df = pd.DataFrame(
        {
            'sample_index': np.arange(len(maternal_signal), dtype=int),
            'time_sec': np.arange(len(maternal_signal), dtype=float) / float(fs),
            'maternal_ecg': maternal_signal,
            'fetal_ecg': fetal_signal,
            'maternal_component_index': int(maternal_idx),
            'fetal_component_index': int(fetal_idx),
            'maternal_component_freq_hz': float(comp_freq[maternal_idx]),
            'fetal_component_freq_hz': float(comp_freq[fetal_idx]),
        }
    )
    pipeline.store_dataframe_in_database(sep_df, 'separated_components', db_path=db_path, if_exists='replace')

    extractor = HRVExtractor(sampling_rate=fs)
    win = fs * args.window_sec

    rows = []
    for start in range(0, len(maternal_signal) - win + 1, win):
        m_chunk = maternal_signal[start:start + win]
        f_chunk = fetal_signal[start:start + win]
        mat = extractor.extract_features(m_chunk)
        fet = extractor.extract_features(f_chunk)
        rows.append(
            {
                'window_start': int(start),
                'window_end': int(start + win),
                'mat_heart_rate_mean': mat['heart_rate_mean'],
                'mat_rmssd': mat['rmssd'],
                'mat_pnn50': mat['pnn50'],
                'fet_heart_rate_mean': fet['heart_rate_mean'],
                'fet_rmssd': fet['rmssd'],
                'fet_pnn50': fet['pnn50'],
                'mat_num_beats': mat['num_beats'],
                'fet_num_beats': fet['num_beats'],
            }
        )

    features_df = pd.DataFrame(rows).dropna()
    if len(features_df) < 15:
        raise ValueError(
            f'Only {len(features_df)} valid feature windows were extracted. '
            'Use a longer record or smaller window size.'
        )

    features_df['target'] = _build_target_labels(features_df)
    pipeline.store_dataframe_in_database(features_df, 'hrv_feature_matrix', db_path=db_path, if_exists='replace')

    feature_cols = [
        'mat_heart_rate_mean',
        'mat_rmssd',
        'mat_pnn50',
        'fet_heart_rate_mean',
        'fet_rmssd',
        'fet_pnn50',
    ]
    X = features_df[feature_cols]
    y = features_df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = CognitiveStateClassifier(model_type='nn')
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    holdout_accuracy = accuracy_score(y_test, y_pred)
    holdout_f1_weighted = f1_score(y_test, y_pred, average='weighted')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    estimator = CognitiveStateClassifier(model_type='nn').model
    cv_f1_scores = cross_val_score(estimator, X, y, cv=cv, scoring='f1_weighted')
    cv_acc_scores = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy')

    model_path = artifacts_dir / 'best_maturation_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    report = pd.DataFrame(
        [
            {
                'data_source': f'physionet:{args.database}/{args.record}',
                'signal_column_primary': signal_column,
                'signal_column_secondary': secondary_column or 'synthetic_shifted_copy',
                'db_table_physio': ingest_info['table_name'],
                'db_table_genomic': 'genomic_data',
                'db_table_behavioral': 'behavioral_data',
                'db_table_separated': 'separated_components',
                'db_table_features': 'hrv_feature_matrix',
                'samples_raw': int(len(raw_df)),
                'samples_genomic': int(len(genomic_df)),
                'samples_behavioral': int(len(behavioral_df)),
                'maternal_component_freq_hz': float(comp_freq[maternal_idx]),
                'fetal_component_freq_hz': float(comp_freq[fetal_idx]),
                'feature_windows': int(len(features_df)),
                'model': 'NeuralNetwork',
                'holdout_accuracy': float(holdout_accuracy),
                'holdout_f1_weighted': float(holdout_f1_weighted),
                'cv_f1_weighted_mean': float(np.mean(cv_f1_scores)),
                'cv_f1_weighted_std': float(np.std(cv_f1_scores)),
                'cv_accuracy_mean': float(np.mean(cv_acc_scores)),
                'cv_accuracy_std': float(np.std(cv_acc_scores)),
                'sources': '; '.join([f"{s['database']}/{s['record']} ({s['rows']} rows)" for s in source_rows]),
            }
        ]
    )

    report_path = artifacts_dir / 'model_evaluation_report.csv'
    report.to_csv(report_path, index=False)

    print('Downloaded and ingested physiological sources:')
    for src in source_rows:
        print(f"  - {src['database']}/{src['record']}: {src['rows']} rows")
    print(f'Total physiological rows (merged): {len(raw_df)}')
    print(f'Primary signal column: {signal_column}')
    print(f'Secondary signal column: {secondary_column or "synthetic_shifted_copy"}')
    print(f'Maternal component frequency (Hz): {comp_freq[maternal_idx]:.4f}')
    print(f'Fetal component frequency (Hz): {comp_freq[fetal_idx]:.4f}')
    print(f'Loaded genomic rows: {len(genomic_df)}')
    print(f'Loaded behavioral rows: {len(behavioral_df)}')
    print(f'Extracted feature windows: {len(features_df)}')
    print(f'Holdout Accuracy: {holdout_accuracy:.4f}')
    print(f'Holdout F1-weighted: {holdout_f1_weighted:.4f}')
    print(f'Saved model: {model_path}')
    print(f'Saved report: {report_path}')


if __name__ == '__main__':
    main()
