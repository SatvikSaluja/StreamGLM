import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from streamglm.matched_benchmark import worker


class MatchedBenchmarkTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('nemos'), 'install streamglm[validation]')
    def test_float64_objective_and_iteration_limit_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(40)
            counts = rng.poisson(.3, (200, 3)).astype(np.int32)
            basis = np.array([[.8, .2], [.2, .8]])
            np.savez(root/'input.npz', counts=counts, basis=basis)
            worker(root/'input.npz', root/'fit', 'nemos_common', max_iter=1)
            result = json.loads((root/'fit.json').read_text())
            y = counts[2:160].astype(float)
            mean = y.mean(axis=0)
            expected = np.sum(mean-y*np.log(mean))/len(y)
            self.assertAlmostEqual(result['initial_objective'], expected, places=12)
            self.assertFalse(result['optimizer_success'])
            self.assertFalse(result['accuracy_gate_passed'])
            self.assertEqual(result['iterations'], 1)

    def test_failed_worker_retains_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                worker(root/'missing.npz', root/'fit', 'streamed')
            result = json.loads((root/'fit.json').read_text())
            self.assertEqual(result['status'], 'FAILED')
            self.assertIn('FileNotFoundError', result['error'])
