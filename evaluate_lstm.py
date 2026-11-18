import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from lstm_data import IMDbDataset, tokenize
from lstm_model import LSTMModel

PROJECT_DIR = Path(__file__).resolve().parent
TEST_PATH = PROJECT_DIR / "data/splits/test.csv"
DEFAULT_CHECKPOINTS = [PROJECT_DIR / f"models/lstm_length_{n}.pt" for n in (50, 100, 250, 500)]
BATCH_SIZE = 64


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test_data(path):
    df = pd.read_csv(path)
    if not {"review", "sentiment"}.issubset(df.columns):
        raise ValueError("The test CSV must contain review and sentiment columns.")
    if df.empty:
        raise ValueError("The test dataset is empty.")
    if df["review"].isna().any() or not df["review"].map(lambda x: isinstance(x, str)).all():
        raise ValueError("Test reviews must be non-missing strings.")
    if not df["sentiment"].isin([0, 1]).all():
        raise ValueError("Test labels must be 0 (negative) or 1 (positive).")
    return df


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config, vocab = checkpoint["config"], checkpoint["vocab"]
    if not isinstance(config["max_length"], int) or config["max_length"] < 1:
        raise ValueError(f"Invalid max_length in {path}")
    if vocab.get("<PAD>") != 0 or vocab.get("<UNK>") != 1:
        raise ValueError(f"Unexpected special token IDs in {path}")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError(f"Vocabulary IDs must be contiguous in {path}")
    model = LSTMModel(vocab_size=len(vocab), **{
        key: config[key] for key in ("embedding_dim", "hidden_dim", "num_layers", "dropout")
    })
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, vocab, config


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    predictions, labels, probabilities = [], [], []
    for inputs, lengths, targets in loader:
        targets = targets.to(device).float()
        logits = model(inputs.to(device), lengths)
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced non-finite logits.")
        total_loss += criterion(logits, targets).item()
        predictions.extend((logits >= 0).long().cpu().tolist())
        labels.extend(targets.long().cpu().tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    if not labels:
        raise ValueError("The test dataset is empty.")
    return {
        "loss": total_loss / len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "macro_f1": f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0),
        "labels": labels, "predictions": predictions, "probabilities": probabilities,
        "classification_report": classification_report(
            labels, predictions, labels=[0, 1], target_names=["negative", "positive"],
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--output-dir", type=Path,
                        help="New output folder; defaults to a timestamped folder in results/.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--limit", type=int, help="Smoke test only: evaluate the first N rows; outputs marked partial.")
    args = parser.parse_args()
    if args.output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        args.output_dir = PROJECT_DIR / "results" / f"lstm_evaluation_{timestamp}"
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")
    return args


def main():
    args = parse_args()
    # Reject existing output directories to preserve previous evaluation evidence.
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new --output-dir; already exists: {args.output_dir}")
    print(f"Output directory: {args.output_dir}", flush=True)
    df = load_test_data(args.test_path)
    full_count = len(df)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}; reviews: {len(df)}/{full_count}", flush=True)
    # Preflight every checkpoint before starting inference or creating outputs.
    configs = []
    for path in args.checkpoints:
        _, _, config = load_checkpoint(path)
        configs.append(config)
    lengths = [c["max_length"] for c in configs]
    if len(lengths) != len(set(lengths)):
        raise ValueError("Supply only one checkpoint per token limit.")
    token_counts = df["review"].map(lambda text: len(tokenize(text)))
    review_ids = df["review"].map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    args.output_dir.mkdir(parents=True)
    summary, reports = [], []
    for path in args.checkpoints:
        model, vocab, config = load_checkpoint(path)
        budget = config["max_length"]
        print(f"Evaluating {path.name} ({budget} tokens)...", flush=True)
        loader = DataLoader(IMDbDataset(df, vocab, budget), batch_size=args.batch_size, shuffle=False)
        result = evaluate(model.to(device), loader, device)
        row = {
            "checkpoint": path.name, "checkpoint_sha256": file_hash(path),
            "max_length": budget, "test_samples": len(df), "partial": len(df) < full_count,
            "loss": result["loss"], "accuracy": result["accuracy"],
            "f1_positive": result["f1"], "macro_f1": result["macro_f1"],
            "reviews_truncated": int((token_counts > budget).sum()),
        }
        predictions = pd.DataFrame({
            "test_row": df.index, "review_id": review_ids.to_numpy(),
            "true_label": result["labels"], "predicted_label": result["predictions"],
            "positive_probability": result["probabilities"],
            "original_token_count": token_counts.to_numpy(),
            "available_token_count": token_counts.clip(upper=budget).to_numpy(),
            "encoded_length": token_counts.clip(lower=1, upper=budget).to_numpy(),
            "max_length": budget,
        })
        predictions.to_csv(args.output_dir / f"predictions_{budget}.csv", index=False)
        summary.append(row)
        reports.append({**row, "checkpoint_path": str(path.resolve()), "config": config,
                        "vocabulary_size": len(vocab),
                        "training_provenance": "Epoch and run metadata absent from checkpoint; not inferred.",
                        "classification_report": result["classification_report"],
                        "confusion_matrix": result["confusion_matrix"]})
        pd.DataFrame(summary).sort_values("max_length").to_csv(args.output_dir / "summary.csv", index=False)
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "test_path": str(args.test_path.resolve()), "test_sha256": file_hash(args.test_path),
            "full_test_samples": full_count, "evaluated_samples": len(df),
            "partial": len(df) < full_count, "device": str(device), "batch_size": args.batch_size,
            "threshold": 0.5, "class_order": ["negative", "positive"],
            "tokenization": "lstm_data.tokenize: lowercase words and individual punctuation; first N tokens",
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
            "source_sha256": {name: file_hash(PROJECT_DIR / name) for name in
                              ("evaluate_lstm.py", "lstm_data.py", "lstm_model.py")},
            "runs": reports,
        }
        (args.output_dir / "evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Accuracy: {result['accuracy']:.4f}; macro F1: {result['macro_f1']:.4f}", flush=True)
    print(f"Saved evaluation to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
