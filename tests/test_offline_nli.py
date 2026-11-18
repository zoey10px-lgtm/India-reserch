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
