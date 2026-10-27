
import re
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text):
    # Convert to lowercase and extract words
    return re.findall(r"\b\w+\b|[^\w\s]", str(text).lower())


def build_vocab(texts, max_vocab_size=20000):
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for word, _ in counter.most_common(max_vocab_size - 2):
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_length):
    tokens = tokenize(text)[:max_length]

    token_ids = [
        vocab.get(token, vocab[UNK_TOKEN])
        for token in tokens
    ]

    # LSTM requires at least one token
    if not token_ids:
        token_ids = [vocab[UNK_TOKEN]]

    length = len(token_ids)

    # Pad to fixed length
    token_ids += [vocab[PAD_TOKEN]] * (max_length - length)

    return token_ids, length


class IMDbDataset(Dataset):
    def __init__(self, dataframe, vocab, max_length):
        self.texts = dataframe["review"].tolist()
        self.labels = dataframe["sentiment"].tolist()
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids, length = encode_text(
            self.texts[idx],
            self.vocab,
            self.max_length
        )

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


def load_data(
    train_path,
    val_path,
    batch_size=64,
    max_length=250,
    max_vocab_size=20000
):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Build vocabulary using training data only
    vocab = build_vocab(
        train_df["review"],
        max_vocab_size=max_vocab_size
    )

    train_dataset = IMDbDataset(train_df, vocab, max_length)
    val_dataset = IMDbDataset(val_df, vocab, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, vocab


if __name__ == "__main__":
    train_loader, val_loader, vocab = load_data(
        train_path="data/splits/train.csv",
        val_path="data/splits/val.csv"
    )

    inputs, lengths, labels = next(iter(train_loader))

    print("Vocabulary size:", len(vocab))
    print("Inputs shape:", inputs.shape)
    print("Lengths shape:", lengths.shape)
    print("Labels shape:", labels.shape)
    print("First review length:", lengths[0].item())
    print("First review label:", labels[0].item())
