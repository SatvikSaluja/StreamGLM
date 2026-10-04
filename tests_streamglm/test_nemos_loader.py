import importlib.util
import unittest
import numpy as np
import jax

from streamglm.data import Recording
from streamglm.integration import NeMoSRecordingLoader


@unittest.skipUnless(importlib.util.find_spec('nemos'), 'install validation extra')
class LoaderTests(unittest.TestCase):
    def test_gap_safe_complete_history_and_tail_coverage(self):
        import nemos as nmo
        jax.config.update('jax_enable_x64', True)
        rng = np.random.default_rng(32)
        counts = rng.poisson(.4, (86, 4))
        counts[36] = 12  # Must never leak across the epoch boundary.
        basis = rng.normal(size=(5, 2))
        recording = Recording(counts, [37, 49])
        loader = NeMoSRecordingLoader(recording, basis, 11)
        expected = np.concatenate([np.asarray(nmo.convolve.create_convolutional_predictor(
            basis, epoch, shift=True))[5:].reshape(len(epoch)-5, -1)
            for epoch in (counts[:37], counts[37:])])
        expected_y = np.concatenate([counts[5:37], counts[42:]])
        for _ in range(2):
            batches = list(loader)
            np.testing.assert_allclose(np.concatenate([x for x, _ in batches]), expected, atol=1e-12)
            np.testing.assert_array_equal(np.concatenate([y for _, y in batches]), expected_y)
        self.assertEqual(loader.n_samples, 76)
        self.assertLessEqual(max(len(y) for _, y in batches), 11)
        np.testing.assert_array_equal(loader.sample_batch()[0], batches[0][0])
