import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3
    ):
        super().__init__()

        # Convert word IDs into dense vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Binary classification output
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        # Ignore padding tokens when processing sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        # Last layer's final hidden state
        output = hidden[-1]

        output = self.dropout(output)
        output = self.fc(output)

        return output.squeeze(-1)

if __name__ == "__main__":
        model = LSTMModel(vocab_size=20000)

        # 3 reviews, each padded to 5 tokens
        x = torch.tensor([
            [12, 45, 67, 89, 0],
            [23, 56, 0, 0, 0],
            [34, 78, 91, 15, 22]
        ])

        lengths = torch.tensor([4, 2, 5])

        output = model(x, lengths)

        print("Model output:", output)
        print("Output shape:", output.shape)

