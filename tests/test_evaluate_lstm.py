"""Checks for metric correctness and inference behavior, independent of trained weights."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluate_lstm import evaluate, load_test_data
from lstm_data import encode_text


class FixedLogits(torch.nn.Module):
    def forward(self, inputs, lengths):
        return inputs[:, 0].float()


class EvaluationTests(unittest.TestCase):
    def test_metrics_threshold_and_partial_batch(self):
        # Uneven final batch; known confusion matrix [[1, 1], [0, 2]].
        loader = DataLoader(TensorDataset(
            torch.tensor([[-2], [0], [2], [1]]), torch.ones(4, dtype=torch.long),
            torch.tensor([0., 1., 0., 1.])), batch_size=3)
        model = FixedLogits()
        model.train()
        result = evaluate(model, loader, torch.device("cpu"))
        self.assertFalse(model.training)
        self.assertEqual(result["predictions"], [0, 1, 1, 1])
        self.assertEqual(result["labels"], [0, 1, 0, 1])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result["accuracy"], .75)
        self.assertAlmostEqual(result["f1"], .8)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([-2., 0., 2., 1.]), torch.tensor([0., 1., 0., 1.])).item()
        self.assertAlmostEqual(result["loss"], expected, places=6)
        self.assertEqual(result["probabilities"][1], .5)

    def test_checkpoint_token_budget_controls_prefix(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "good": 2, "but": 3, "bad": 4, "!": 5}
        self.assertEqual(encode_text("Good but bad!", vocab, 2), ([2, 3], 2))
        self.assertEqual(encode_text("Good but bad!", vocab, 5), ([2, 3, 4, 5, 0], 4))
        self.assertEqual(encode_text("", vocab, 2), ([1, 0], 1))

    def test_reject_invalid_test_labels_and_missing_reviews(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            for data in ({"review": ["a"], "sentiment": [2]},
                         {"review": [None], "sentiment": [1]}):
                pd.DataFrame(data).to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_test_data(path)


if __name__ == "__main__":
    unittest.main()
