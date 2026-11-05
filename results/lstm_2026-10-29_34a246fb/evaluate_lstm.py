"""Evaluate the saved LSTM on the held-out IMDb test split."""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from lstm_data import IMDbDataset
from lstm_model import LSTMModel


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "best_lstm.pt"
TEST_PATH = PROJECT_DIR / "data" / "splits" / "test.csv"
BATCH_SIZE = 64


@torch.no_grad()
def evaluate(model, loader, device):
    # Disable dropout and gradient tracking during evaluation.
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for inputs, lengths, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float()

        # Sequence lengths stay on CPU for pack_padded_sequence.
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        # A logit of zero corresponds to a positive probability of 0.5.
        predictions = (logits >= 0).long()
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().long().tolist())

    if not all_labels:
        raise ValueError("The test dataset is empty.")

    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_predictions),
        # Match training: binary F1 with positive sentiment as class 1.
        "f1": f1_score(all_labels, all_predictions, zero_division=0),
        "labels": all_labels,
        "predictions": all_predictions,
    }


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run train_lstm.py before evaluating."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Reuse the training vocabulary and configuration without fitting on test data.
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    vocab = checkpoint["vocab"]
    config = checkpoint["config"]

    model = LSTMModel(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_df = pd.read_csv(TEST_PATH)
    if not {"review", "sentiment"}.issubset(test_df.columns):
        raise ValueError("The test CSV must contain review and sentiment columns.")
    if test_df.empty:
        raise ValueError("The test dataset is empty.")
    if test_df["review"].isna().any():
        raise ValueError("The test dataset contains missing reviews.")
    if not test_df["sentiment"].isin([0, 1]).all():
        raise ValueError("Test labels must be 0 (negative) or 1 (positive).")

    test_dataset = IMDbDataset(test_df, vocab, config["max_length"])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    results = evaluate(model, test_loader, device)

    print("Test reviews:", len(test_dataset))
    print("Maximum tokens per review:", config["max_length"])
    print(f"Test loss: {results['loss']:.4f}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Test F1 (positive class): {results['f1']:.4f}")

    print("\nClassification report:")
    print(classification_report(
        results["labels"],
        results["predictions"],
        labels=[0, 1],
        target_names=["negative", "positive"],
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print("Class order: negative (0), positive (1)")
    print(confusion_matrix(results["labels"], results["predictions"], labels=[0, 1]))


if __name__ == "__main__":
    main()
