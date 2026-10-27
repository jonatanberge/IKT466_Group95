
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from lstm_model import LSTMModel
from lstm_data import load_data


# Hyperparameters
BATCH_SIZE = 64
MAX_LENGTH = 250
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 10
PATIENCE = 3


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    all_predictions = []
    all_labels = []

    for inputs, lengths, labels in loader:
        inputs = inputs.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device).float()

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs, lengths)

        # Calculate loss
        loss = criterion(logits, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        # Convert logits to binary predictions
        predictions = (logits >= 0).long()

        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(labels.cpu().long().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return avg_loss, accuracy, f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    all_predictions = []
    all_labels = []

    for inputs, lengths, labels in loader:
        inputs = inputs.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device).float()

        logits = model(inputs, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)

        predictions = (logits >= 0).long()

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().long().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return avg_loss, accuracy, f1


def main():
    # Reproducibility
    torch.manual_seed(42)

    # Select CPU or GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Load preprocessed data
    train_loader, val_loader, vocab = load_data(
        train_path="data/splits/train.csv",
        val_path="data/splits/val.csv",
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH
    )

    # Initialize model
    model = LSTMModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    patience_counter = 0

    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(
            f"Train - Loss: {train_loss:.4f}, "
            f"Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}"
        )
        print(
            f"Val   - Loss: {val_loss:.4f}, "
            f"Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "embedding_dim": EMBEDDING_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "max_length": MAX_LENGTH
                    }
                },
                output_dir / "best_lstm.pt"
            )

            print("Best model saved!")

        else:
            patience_counter += 1

            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nTraining finished!")


if __name__ == "__main__":
    main()
