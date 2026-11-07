import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from lstm_model import LSTMModel
from lstm_data import load_data
from train_lstm import (
    train_one_epoch,
    validate,
    BATCH_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE
)


# Experiment settings

MAX_LENGTHS = [100, 500]

SEED = 42

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment(max_length, device):
    print(f"\n{'=' * 50}", flush=True)
    print(f"Training LSTM with {max_length} tokens", flush=True)
    print(f"{'=' * 50}", flush=True)

    # Same random seed for reproducibility
    torch.manual_seed(SEED)

    train_loader, val_loader, vocab = load_data(
        train_path="data/splits/train.csv",
        val_path="data/splits/val.csv",
        batch_size=BATCH_SIZE,
        max_length=max_length
    )

    model = LSTMModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f}, "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, "
            f"Val acc: {val_acc:.4f}, "
            f"Val F1: {val_f1:.4f}",
            flush=True
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            best_metrics = {
                "max_length": max_length,
                "best_epoch": best_epoch,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "val_loss": val_loss
            }

            # Save model and vocabulary
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "embedding_dim": EMBEDDING_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "max_length": max_length
                    }
                },
                MODEL_DIR / f"lstm_length_{max_length}.pt"
            )

            print("Best model saved!", flush=True)

        else:
            patience_counter += 1

            if patience_counter >= PATIENCE:
                print("Early stopping triggered.", flush=True)
                break

    elapsed = time.perf_counter() - start_time
    best_metrics["training_minutes"] = round(elapsed / 60, 2)

    print(
        f"Finished {max_length} tokens in "
        f"{elapsed / 60:.2f} minutes",
        flush=True
    )

    return best_metrics


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device, flush=True)

    results = []

    for max_length in MAX_LENGTHS:
        result = run_experiment(max_length, device)
        results.append(result)

        # Save results after each completed experiment
        output_path = RESULTS_DIR / "review_length_results.csv"

        with open(output_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "max_length",
                    "best_epoch",
                    "train_accuracy",
                    "val_accuracy",
                    "val_f1",
                    "val_loss",
                    "training_minutes"
                ]
            )

            writer.writeheader()
            writer.writerows(results)

    print("\nAll review length experiments finished!")


if __name__ == "__main__":
    main()
