"""Validate and analyse saved LSTM predictions without repeating inference."""
import argparse
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path

# Keep matplotlib's cache in a writable location and support headless execution.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ikt466_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from evaluate_lstm import PROJECT_DIR, file_hash, load_test_data
from lstm_data import tokenize


def binary_metrics(labels, predictions):
    """Return accuracy and macro F1 for each model column (both labels always included)."""
    labels = np.asarray(labels)[:, None]
    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    tp = ((labels == 1) & (predictions == 1)).sum(axis=0)
    tn = ((labels == 0) & (predictions == 0)).sum(axis=0)
    fp = ((labels == 0) & (predictions == 1)).sum(axis=0)
    fn = ((labels == 1) & (predictions == 0)).sum(axis=0)
    def divide(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)
    return np.stack(((tp + tn) / len(labels),
                     (divide(2 * tp, 2 * tp + fp + fn) + divide(2 * tn, 2 * tn + fp + fn)) / 2), axis=1)


def paired_bootstrap(labels, predictions, iterations=5000, seed=42):
    """Same resampled review rows for every model; ordinary percentile bootstrap."""
    rng = np.random.default_rng(seed)
    samples = np.empty((iterations, predictions.shape[1], 2))
    for i in range(iterations):
        indices = rng.integers(0, len(labels), size=len(labels))
        samples[i] = binary_metrics(labels[indices], predictions[indices])
    return samples


def load_results(input_dir, test_path):
    metadata = json.loads((input_dir / "evaluation.json").read_text())
    if metadata["partial"]:
        raise ValueError("Final analysis requires a full test evaluation, not a partial run.")
    if file_hash(test_path) != metadata["test_sha256"]:
        raise ValueError("Test CSV hash differs from the evaluated input.")
    df = load_test_data(test_path)
    summary = pd.read_csv(input_dir / "summary.csv").sort_values("max_length").reset_index(drop=True)
    if summary.max_length.tolist() != [50, 100, 250, 500]:
        raise ValueError("Expected exactly the four planned token budgets: 50, 100, 250, 500.")
    if metadata["evaluated_samples"] != len(df):
        raise ValueError("Evaluation sample count differs from source data.")
    ids = df.review.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()).to_numpy()
    counts = df.review.map(lambda text: len(tokenize(text))).to_numpy()
    predictions = []
    for row in summary.to_dict("records"):
        budget = row["max_length"]
        frame = pd.read_csv(input_dir / f"predictions_{budget}.csv")
        if len(frame) != len(df) or not np.array_equal(frame.test_row, np.arange(len(df))):
            raise ValueError(f"Missing, reordered, or duplicated rows for {budget} tokens.")
        for actual, expected in ((frame.review_id, ids), (frame.true_label, df.sentiment),
                                 (frame.original_token_count, counts),
                                 (frame.available_token_count, np.minimum(counts, budget)),
                                 (frame.encoded_length, np.clip(counts, 1, budget))):
            if not np.array_equal(actual, expected):
                raise ValueError(f"Prediction alignment or token counts invalid for {budget} tokens.")
        if not frame.predicted_label.isin([0, 1]).all() or not frame.positive_probability.between(0, 1).all():
            raise ValueError(f"Invalid predictions/probabilities for {budget} tokens.")
        if not (frame.max_length == budget).all() or row['test_samples'] != len(df) or row['partial']:
            raise ValueError(f"Invalid budget or sample metadata for {budget} tokens.")
        expected_metrics = [accuracy_score(df.sentiment, frame.predicted_label),
                            f1_score(df.sentiment, frame.predicted_label, zero_division=0),
                            f1_score(df.sentiment, frame.predicted_label, labels=[0, 1], average='macro', zero_division=0)]
        if not np.allclose(expected_metrics, [row['accuracy'], row['f1_positive'], row['macro_f1']], atol=1e-12, rtol=0):
            raise ValueError(f"Saved summary differs from predictions for {budget} tokens.")
        run = next(r for r in metadata['runs'] if r['config']['max_length'] == budget)
        if run['checkpoint_sha256'] != row['checkpoint_sha256'] or run['confusion_matrix'] != confusion_matrix(df.sentiment, frame.predicted_label, labels=[0, 1]).tolist():
            raise ValueError(f"JSON report differs from summary/predictions for {budget} tokens.")
        predictions.append(frame.predicted_label.to_numpy())
    return df, counts, summary, np.column_stack(predictions), metadata


def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=220, bbox_inches="tight")
    plt.close(fig)


def figures(output, summary, labels, predictions):
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, label, offset in (("accuracy", "Accuracy", -4), ("macro_f1", "Macro F1", 4)):
        values = summary[metric].to_numpy()
        low, high = summary[f'{metric}_ci_low'].to_numpy(), summary[f'{metric}_ci_high'].to_numpy()
        ax.errorbar(summary.max_length + offset, values, yerr=[values - low, high - values], marker='o', capsize=4, label=label)
    ax.set(xlabel='Maximum tokens per review', ylabel='Score', xticks=summary.max_length,
           title='LSTM test performance by available text\n95% paired-review bootstrap intervals')
    ax.set_ylim(.7, .92)
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    save_figure(fig, output / 'performance')
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    matrices = [confusion_matrix(labels, predictions[:, i], labels=[0, 1]) for i in range(4)]
    for ax, budget, matrix in zip(axes.flat, summary.max_length, matrices):
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(m.max() for m in matrices))
        for (r, c), value in np.ndenumerate(matrix):
            ax.text(c, r, str(value), ha='center', va='center', color='white' if value > max(m.max() for m in matrices) / 2 else 'black')
        ax.set(title=f'{budget} tokens', xlabel='Predicted sentiment', ylabel='Actual sentiment',
               xticks=[0, 1], yticks=[0, 1], xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    fig.suptitle('LSTM test confusion matrices (counts; shared colour scale)')
    save_figure(fig, output / 'confusion_matrices')


def error_exports(output, df, counts, budgets, predictions):
    labels = df.sentiment.to_numpy()
    transitions = []
    correct = predictions == labels[:, None]
    for a, b in itertools.combinations(range(len(budgets)), 2):
        for group, mask in [('all', np.ones(len(labels), dtype=bool)),
                            ('additional_text_available', counts > budgets[a]),
                            ('same_input_text', counts <= budgets[a])]:
            transitions.append({
                'from_tokens': budgets[a], 'to_tokens': budgets[b], 'group': group,
                'reviews': int(mask.sum()),
                'wrong_to_correct': int((~correct[:, a] & correct[:, b] & mask).sum()),
                'correct_to_wrong': int((correct[:, a] & ~correct[:, b] & mask).sum()),
                'both_correct': int((correct[:, a] & correct[:, b] & mask).sum()),
                'both_wrong': int((~correct[:, a] & ~correct[:, b] & mask).sum()),
            })
    pd.DataFrame(transitions).to_csv(output / 'prediction_transitions.csv', index=False)
    # Select the first two eligible source rows per category: reproducible, not ranked by confidence.
    categories = {
        'persistent_error': (~correct).all(axis=1),
        'corrected_50_to_250': ~correct[:, 0] & correct[:, 2] & (counts > 50),
        'regressed_250_to_500': correct[:, 2] & ~correct[:, 3] & (counts > 250),
        'negation_cue_error': (~correct).any(axis=1) & df.review.str.contains(r"\b(?:not|never|no|cannot)\b|n't", case=False, regex=True).to_numpy(),
        'contrast_cue_error': (~correct).any(axis=1) & df.review.str.contains(r'\b(?:but|although|however)\b', case=False, regex=True).to_numpy(),
    }
    examples = []
    for category, mask in categories.items():
        for idx in np.flatnonzero(mask)[:2]:
            tokens = tokenize(df.review.iloc[idx])
            item = {'category': category, 'test_row': int(idx), 'true_label': int(labels[idx]),
                    'original_token_count': int(counts[idx]), 'review': df.review.iloc[idx]}
            for i, budget in enumerate(budgets):
                item[f'predicted_{budget}'] = int(predictions[idx, i])
                item[f'input_tokens_{budget}'] = tokens[:budget]
            examples.append(item)
    (output / 'error_examples.json').write_text(json.dumps(examples, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--test-path', type=Path, default=PROJECT_DIR / 'data/splits/test.csv')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--bootstrap-iterations', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1000:
        parser.error('Use at least 1000 bootstrap iterations for final analysis.')
    output = args.output_dir or args.input_dir / 'analysis'
    if output.exists():
        raise FileExistsError(f'Choose a new output folder: {output}')
    df, counts, summary, predictions, metadata = load_results(args.input_dir, args.test_path)
    labels = df.sentiment.to_numpy()
    print(f'Validated {len(df)} reviews across all four budgets. Bootstrapping...', flush=True)
    samples = paired_bootstrap(labels, predictions, args.bootstrap_iterations, args.seed)
    points = binary_metrics(labels, predictions)
    intervals = np.quantile(samples, [.025, .975], axis=0)
    for j, metric in enumerate(('accuracy', 'macro_f1')):
        summary[f'{metric}_ci_low'] = intervals[0, :, j]
        summary[f'{metric}_ci_high'] = intervals[1, :, j]
    comparisons = []
    budgets = summary.max_length.tolist()
    for a, b in itertools.combinations(range(4), 2):
        ci = np.quantile(samples[:, b] - samples[:, a], [.025, .975], axis=0)
        for j, metric in enumerate(('accuracy', 'macro_f1')):
            comparisons.append({'from_tokens': budgets[a], 'to_tokens': budgets[b], 'metric': metric,
                                'difference': points[b, j] - points[a, j], 'ci_low': ci[0, j], 'ci_high': ci[1, j]})
    output.mkdir(parents=True)
    summary.to_csv(output / 'summary_with_intervals.csv', index=False)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / 'paired_comparisons.csv', index=False)
    error_exports(output, df, counts, budgets, predictions)
    figures(output, summary, labels, predictions)
    manifest = {
        'bootstrap_iterations': args.bootstrap_iterations, 'seed': args.seed,
        'method': 'Ordinary paired review bootstrap; 95% percentile intervals; same sampled rows across all models.',
        'interval_scope': 'Fixed trained models; test-review sampling only; no retraining variability. Pairwise intervals are unadjusted for multiple comparisons.',
        'source_files_sha256': {p.name: file_hash(p) for p in [args.input_dir / 'summary.csv', args.input_dir / 'evaluation.json'] + sorted(args.input_dir.glob('predictions_*.csv'))},
        'analysis_source_sha256': file_hash(Path(__file__)),
        'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
    }
    (output / 'analysis_metadata.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Analysis saved to {output}', flush=True)


if __name__ == '__main__':
    main()
