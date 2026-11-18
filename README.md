# IKT466_Group95

## LSTM evaluation

Evaluate the saved 50-, 100-, 250-, and 500-token models on the held-out test split:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-evaluation.txt
.venv/bin/python evaluate_lstm.py
```

The final results are in `results/lstm_evaluation/`. New evaluations create a
separate timestamped folder. Use `--output-dir` to choose a new destination;
existing folders are preserved. `--checkpoints`, `--test-path`, `--batch-size`,
and `--device` provide overrides. `--limit N` allows a quick partial check, whose
results must not be used for final reporting.

### Saved files

- `summary.csv`: loss, accuracy, positive-class F1, macro F1, and truncation counts.
- `evaluation.json`: per-class metrics, confusion matrices, model configurations,
  software versions, and source/checkpoint hashes.
- `predictions_N.csv`: per-review predictions and probabilities, token counts,
  source CSV row (`test_row`), and stable review identifier.
- `checkpoint_audit.json`: checkpoint identities and training-vocabulary checks.
- `analysis/`: LSTM confidence intervals, paired differences, prediction transitions,
  selected error examples, metadata, and two PNG figures.
- `model_comparison/`: matched Naive Bayes predictions, model-comparison tables,
  metadata, and a comparison PNG.

Discussion and interpretation belong in the external project report; the scripts
save data and figures without generating Markdown reports or duplicate PDF copies.
The error examples include the original text and exact normalized input tokens.
Negation/contrast categories use lexical cues, not verified explanations of errors.

### Reproduce analysis and comparison

```sh
.venv/bin/python analyze_lstm.py results/lstm_evaluation --output-dir /tmp/lstm_analysis
.venv/bin/python compare_sentiment_models.py results/lstm_evaluation --output-dir /tmp/lstm_comparison
.venv/bin/python -m unittest discover -s tests -v
```

Choose fresh output folders when rerunning these commands. Both analysis scripts
use 5,000 paired bootstrap draws (seed 42). Intervals estimate test-review sampling
uncertainty for fixed weights, exclude training-seed variation, and are unadjusted
for multiple comparisons. Metadata records settings and input/source hashes.

### Experimental setup

Accuracy and **macro F1** are the main comparison metrics. Positive sentiment is
class 1; the LSTM threshold is 0.5. Each checkpoint supplies its training vocabulary
and configuration. Nothing is fitted on test data.

Both model families receive the same first N lowercase word/punctuation tokens.
The matched Naive Bayes baseline uses training-only TF-IDF (10,000 features) and
MultinomialNB(alpha=1.0), preserving punctuation and single-character tokens.
`models/naive_bayes.py` is the original, separate validation analysis.

Each token budget uses a separately trained LSTM, so differences reflect training
and evaluation under that budget. The experiment varies text per review, not
training-set size. HTML remains in the existing preprocessing.

Checkpoint epochs and full training-run metadata are unavailable. The audit shows
that `models/best_lstm.pt` matches the archived interrupted one-epoch model;
`models/lstm_length_250.pt` matches the archived `previous_best_lstm.pt`.
Evaluation uses the four length-specific checkpoints. Vocabulary agreement does
not establish complete historical training provenance.

The older `results/lstm_2026-10-29_34a246fb/` archive and
`results/review_length_results.csv` predate this evaluation; the latter contains
validation metrics, not final test results.
