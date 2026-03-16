import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from data_pipeline import DataPipeline
from feature_extraction import HRVExtractor
from model import CognitiveStateClassifier
from preprocessing import ECGPreprocessor
from signal_separation import ComponentAnalyzer, SignalSeparator


def _auto_pick_signal_column(df: pd.DataFrame) -> str:
    excluded = {
        'sample_index',
        'time_sec',
        'record_name',
        'database',
        'sampling_rate',
    }
    numeric_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError('No numeric ECG signal column found in ingested PhysioNet table.')
    return numeric_cols[0]


def _numeric_signal_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        'sample_index',
        'time_sec',
        'record_name',
        'database',
        'sampling_rate',
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
    parser.add_argument('--database', default='mitdb', help='PhysioNet dataset name (e.g., mitdb).')
    parser.add_argument('--record', default='100', help='Record identifier in the selected PhysioNet dataset.')
    parser.add_argument('--signal-column', default=None, help='Specific ECG column to use after ingestion.')
    parser.add_argument('--window-sec', type=int, default=10, help='Window length in seconds for HRV extraction.')
    parser.add_argument('--genomic-csv', default=None, help='Optional genomic CSV path to load into genomic_data table.')
    parser.add_argument('--behavioral-csv', default=None, help='Optional behavioral CSV path to load into behavioral_data table.')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    db_path = root / 'data' / 'processed' / 'neuro_genomic.db'
    artifacts_dir = root / 'results' / 'models'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DataPipeline(data_dir=root / 'data', db_path=db_path)

    ingest_info = pipeline.ingest_physionet_record_to_database(
        database=args.database,
        record_name=args.record,
        table_name='physio_data',
        db_path=db_path,
        if_exists='replace',
    )

    raw_df = pipeline.load_table_from_database('physio_data', db_path=db_path)

    if args.genomic_csv:
        genomic_df = pd.read_csv(args.genomic_csv)
    else:
        genomic_df = _build_default_genomic_table(max(20, len(raw_df) // 1000))
    pipeline.store_dataframe_in_database(genomic_df, 'genomic_data', db_path=db_path, if_exists='replace')

    if args.behavioral_csv:
        behavioral_df = pd.read_csv(args.behavioral_csv)
    else:
        behavioral_df = _build_default_behavioral_table(max(20, len(raw_df) // 1000))
    pipeline.store_dataframe_in_database(behavioral_df, 'behavioral_data', db_path=db_path, if_exists='replace')

    numeric_cols = _numeric_signal_columns(raw_df)
    signal_column = args.signal_column or _auto_pick_signal_column(raw_df)
    if signal_column not in numeric_cols:
        raise ValueError(f'Signal column {signal_column} is not present in downloaded PhysioNet record.')

    secondary_candidates = [c for c in numeric_cols if c != signal_column]
    secondary_column = secondary_candidates[0] if secondary_candidates else None

    fs = int(float(raw_df['sampling_rate'].iloc[0])) if 'sampling_rate' in raw_df.columns else 360

    primary_signal = raw_df[signal_column].astype(float).to_numpy()
    if secondary_column is not None:
        secondary_signal = raw_df[secondary_column].astype(float).to_numpy()
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
            }
        ]
    )

    report_path = artifacts_dir / 'model_evaluation_report.csv'
    report.to_csv(report_path, index=False)

    print(f'Downloaded and ingested physiological rows: {len(raw_df)}')
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
