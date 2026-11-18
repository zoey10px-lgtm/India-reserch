import unittest

from research.offline_nli import OfflineNLI


class BinaryNliTests(unittest.TestCase):
    def test_non_entailment_alone_is_neutral(self):
        model = OfflineNLI.__new__(OfflineNLI)
        model.binary = True
        model.labels = {"entailment": 0, "not_entailment": 1}

        def probabilities(pairs, batch_size):
            self.assertEqual(pairs[2:], [
                ("missing", "It is false that the device was tested."),
                ("denial", "It is false that the device was tested."),
            ])
            return [[.05, .95], [.05, .95], [.1, .9], [.96, .04]]

        model._probabilities = probabilities
        neutral, contradicted = model.predict_batch([
            ("missing", "the device was tested."), ("denial", "the device was tested."),
        ])
        self.assertGreater(neutral.neutral, .8)
        self.assertGreater(contradicted.contradiction, .9)

    def test_thresholded_support_checks_denial_for_possible_passes(self):
        model = OfflineNLI.__new__(OfflineNLI)
        model.binary = True
        model.labels = {"entailment": 0, "not_entailment": 1}
        pairs = [(name, "The test was completed.") for name in ("low", "pass", "ambiguous")]

        def probabilities(batch, batch_size):
            if batch == pairs:
                return [[.20, .80], [.90, .10], [.91, .09]]
            self.assertEqual([text for text, _ in batch], ["pass", "ambiguous"])
            return [[.02, .98], [.90, .10]]

        model._probabilities = probabilities
        self.assertEqual(model.passing_support(pairs, .7), [0.0, .9, 0.0])
