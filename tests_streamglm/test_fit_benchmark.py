import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamglm import fit_benchmark


class FitBenchmarkTests(unittest.TestCase):
    def test_failure_is_recorded_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'run'
            with patch('sys.argv', ['benchmark', '--out', str(out)]), \
                    patch.object(fit_benchmark, 'fixture', side_effect=RuntimeError('fixture failed')):
                with self.assertRaisesRegex(RuntimeError, 'fixture failed'):
                    fit_benchmark.main()
                result = json.loads((out/'benchmark.json').read_text())
                self.assertEqual(result['status'], 'FAILED')
                self.assertGreater(result['peak_rss_gib'], 0)
                self.assertGreaterEqual(result['total_seconds'], 0)
                with self.assertRaises(FileExistsError):
                    fit_benchmark.main()
                self.assertEqual(result, json.loads((out/'benchmark.json').read_text()))
