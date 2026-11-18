"""Statistical checks against sklearn and paired-bootstrap invariants."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from analyze_lstm import binary_metrics, paired_bootstrap, load_results


class AnalysisTests(unittest.TestCase):
    def test_metrics_match_sklearn_including_single_class(self):
        for labels, predictions in [([0, 0, 1, 1], [0, 1, 1, 1]),
                                    ([0, 0], [0, 0]), ([1, 1], [0, 0])]:
            values = binary_metrics(labels, predictions)[0]
            self.assertAlmostEqual(values[0], accuracy_score(labels, predictions))
            self.assertAlmostEqual(values[1], f1_score(labels, predictions, labels=[0, 1], average='macro', zero_division=0))

    def test_paired_identical_models_have_zero_difference(self):
        labels = np.array([0, 1, 0, 1, 1])
        predictions = np.column_stack(([0, 1, 1, 1, 0], [0, 1, 1, 1, 0]))
        first = paired_bootstrap(labels, predictions, iterations=100, seed=42)
        second = paired_bootstrap(labels, predictions, iterations=100, seed=42)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:, 1] - first[:, 0], np.zeros((100, 2)))

    def test_accuracy_difference_retains_direction_in_every_draw(self):
        labels = np.array([0, 1, 0, 1])
        predictions = np.column_stack((1 - labels, labels))
        samples = paired_bootstrap(labels, predictions, iterations=100, seed=42)
        np.testing.assert_array_equal(samples[:, 1, 0] - samples[:, 0, 0], np.ones(100))

    def test_final_analysis_rejects_partial_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'evaluation.json').write_text(json.dumps({'partial': True}))
            with self.assertRaisesRegex(ValueError, 'full test evaluation'):
                load_results(folder, folder / 'test.csv')


if __name__ == '__main__':
    unittest.main()
