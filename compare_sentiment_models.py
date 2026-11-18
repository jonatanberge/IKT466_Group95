"""Matched Naive Bayes baseline for the completed LSTM prefix-budget evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import sklearn

from analyze_lstm import binary_metrics, load_results, paired_bootstrap, plt, save_figure
from evaluate_lstm import PROJECT_DIR, file_hash, load_test_data
from lstm_data import tokenize


def identity(tokens):
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lstm_results', type=Path)
    parser.add_argument('--train-path', type=Path, default=PROJECT_DIR / 'data/splits/train.csv')
    parser.add_argument('--test-path', type=Path, default=PROJECT_DIR / 'data/splits/test.csv')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--bootstrap-iterations', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1000:
        parser.error('Use at least 1000 bootstrap iterations.')
    output = args.output_dir or args.lstm_results / 'model_comparison'
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    test, _, summary, lstm_predictions, _ = load_results(args.lstm_results, args.test_path)
    train = load_test_data(args.train_path)
    if set(train.review) & set(test.review):
        raise ValueError('Train and test contain overlapping reviews.')
    # The exact same normalized prefix tokens are available to both model families.
    train_tokens = train.review.map(tokenize)
    test_tokens = test.review.map(tokenize)
    labels = test.sentiment.to_numpy()
    nb_predictions, reports, rows = [], [], []
    output.mkdir(parents=True)
    for budget in summary.max_length:
        print(f'Fitting fixed Naive Bayes baseline: {budget} tokens...', flush=True)
        vectorizer = TfidfVectorizer(analyzer=identity, max_features=10000)
        x_train = vectorizer.fit_transform(train_tokens.map(lambda tokens: tokens[:budget]))
        x_test = vectorizer.transform(test_tokens.map(lambda tokens: tokens[:budget]))
        model = MultinomialNB(alpha=1.0)
        model.fit(x_train, train.sentiment)
        prediction = model.predict(x_test)
        nb_predictions.append(prediction)
        pd.DataFrame({'test_row': np.arange(len(test)), 'true_label': labels,
                      'predicted_label': prediction, 'positive_probability': model.predict_proba(x_test)[:, 1],
                      'max_length': budget}).to_csv(output / f'naive_bayes_predictions_{budget}.csv', index=False)
        reports.append({'max_length': int(budget), 'vocabulary_size': len(vectorizer.vocabulary_),
                        'classification_report': classification_report(labels, prediction, labels=[0, 1], target_names=['negative', 'positive'], output_dict=True, zero_division=0),
                        'confusion_matrix': confusion_matrix(labels, prediction, labels=[0, 1]).tolist()})
    nb_predictions = np.column_stack(nb_predictions)
    combined = np.column_stack((lstm_predictions, nb_predictions))
    points = binary_metrics(labels, combined)
    print('Calculating paired model-family intervals...', flush=True)
    samples = paired_bootstrap(labels, combined, args.bootstrap_iterations, args.seed)
    intervals = np.quantile(samples, [.025, .975], axis=0)
    differences = []
    for i, budget in enumerate(summary.max_length):
        ci = np.quantile(samples[:, i] - samples[:, i + 4], [.025, .975], axis=0)
        for model_name, index in [('LSTM', i), ('Naive Bayes', i + 4)]:
            rows.append({'model': model_name, 'max_length': int(budget), 'test_samples': len(test),
                         **{key: float(value) for j, metric in enumerate(['accuracy', 'macro_f1']) for key, value in
                            [(metric, points[index, j]), (f'{metric}_ci_low', intervals[0, index, j]), (f'{metric}_ci_high', intervals[1, index, j])]}})
        for j, metric in enumerate(['accuracy', 'macro_f1']):
            differences.append({'max_length': int(budget), 'metric': metric, 'lstm_minus_nb': points[i, j] - points[i + 4, j], 'ci_low': ci[0, j], 'ci_high': ci[1, j]})
    pd.DataFrame(rows).to_csv(output / 'model_comparison.csv', index=False)
    pd.DataFrame(differences).to_csv(output / 'paired_model_differences.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    for j, (ax, metric) in enumerate(zip(axes, ['accuracy', 'macro_f1'])):
        for name, start in [('LSTM', 0), ('Naive Bayes', 4)]:
            values = points[start:start+4, j]
            ax.errorbar(summary.max_length, values,
                        yerr=[values - intervals[0, start:start+4, j], intervals[1, start:start+4, j] - values], marker='o', capsize=3, label=name)
        ax.set(title=metric.replace('_', ' ').title(), xlabel='Maximum tokens per review', xticks=summary.max_length, ylim=(.7, .92))
        ax.grid(axis='y', alpha=.25)
        ax.legend()
    axes[0].set_ylabel('Score (95% paired-review bootstrap CI)')
    save_figure(fig, output / 'model_comparison')
    metadata = {'train_sha256': file_hash(args.train_path), 'test_sha256': file_hash(args.test_path),
                'train_samples': len(train), 'test_samples': len(test), 'max_features': 10000, 'alpha': 1.0,
                'feature_definition': 'TF-IDF unigrams over exact lstm_data.tokenize prefix tokens, including punctuation; training-only IDF/vocabulary.',
                'hyperparameters': 'Fixed baseline settings; no test-driven tuning; validation data not used for fitting.',
                'bootstrap_iterations': args.bootstrap_iterations, 'seed': args.seed,
                'interval_scope': 'Fixed models, review-sampling uncertainty; paired percentile intervals, unadjusted for multiple comparisons.',
                'source_sha256': file_hash(Path(__file__)), 'sklearn_version': sklearn.__version__,
                'lstm_input_hashes': {p.name: file_hash(p) for p in sorted(args.lstm_results.glob('predictions_*.csv'))},
                'naive_bayes_reports': reports}
    (output / 'comparison_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved matched comparison to {output}', flush=True)


if __name__ == '__main__':
    main()
