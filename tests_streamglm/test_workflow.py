from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import tempfile
import unittest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from streamglm import Recording, StreamingGLM
from streamglm.adapters import from_spike_times, from_pynapple, from_nwb, load_recording, from_hnn_cache
from streamglm.evaluation import split_recording, score
from streamglm.optim import fit_adam
from streamglm.persistence import save_bundle, load_bundle
from streamglm.reference import features, materialize_weights
from streamglm.cli import main as cli


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.y = np.random.default_rng(4).poisson(.2, (120, 3))
        self.basis = np.array([[.6, .1], [.3, .3], [.1, .6]])

    def test_self_history_dense_and_lowrank_gradient(self):
        data = Recording(self.y, [49, 71])
        x, valid = features(self.y, self.basis, data.epoch_lengths)
        for rank in (None, 0, 2):
            model = StreamingGLM(self.basis, 17, rank, .1, self_history=True)
            p = model.initialize(3)
            p["S"] = jnp.ones((2, 3))*.15
            def objective(q):
                w = q["W"] if rank is None else jnp.einsum("kir,kjr->kij", q["U"], q["V"])
                idx = jnp.arange(3)
                w = w.at[:, idx, idx].set(q["S"])
                eta = jnp.einsum("tki,kij->tj", jnp.asarray(x[valid]), w)+q["b"]
                return jnp.sum(jnp.exp(eta)-self.y[valid]*eta)/valid.sum()+.05*jnp.sum(w*w)
            value, gradient = model.value_and_grad(p, data)
            expected, grad = jax.value_and_grad(objective)(p)
            np.testing.assert_allclose(value, expected, atol=1e-11)
            for k in grad:
                np.testing.assert_allclose(gradient[k], grad[k], atol=1e-11)

    def test_split_is_lazy_and_resets_boundaries(self):
        data = Recording(self.y, [50, 70], bin_s=.01, unit_ids=[7, 8, 9])
        train, val, test = split_recording(data)
        self.assertEqual(train.epoch_lengths, (30, 42))
        self.assertEqual(val.epoch_lengths, (10, 14))
        self.assertEqual(test.epoch_lengths, (10, 14))
        expected = np.concatenate([self.y[:30], self.y[50:92]])
        np.testing.assert_array_equal(train.counts[:], expected)
        selected = train.select([(0, 72)], [2, 0])
        self.assertEqual(selected.unit_ids, [9, 7])
        self.assertEqual(selected.epoch_lengths, (30, 42))
        np.testing.assert_array_equal(selected.counts[28:33], expected[28:33][:, [2, 0]])

    def test_half_open_binning_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"recording"
            data = from_spike_times([[0., .1, .2, 1., 2., 2.99, 3.], [.15]],
                [[0, 1], [2, 3]], .1, path, [10, 11])
            self.assertEqual(data.counts[:, 0].sum(), 5)
            self.assertEqual(data.counts[1, 0], 1)
            self.assertEqual(data.counts[10, 0], 1)
            self.assertEqual(data.epoch_lengths, (10, 10))
            self.assertIsInstance(data.counts, np.memmap)
            with self.assertRaises(FileExistsError):
                from_spike_times([[]], [[0, 1]], .1, path)

    def test_absolute_timestamp_bin_edge(self):
        with tempfile.TemporaryDirectory() as folder:
            data = from_spike_times([[8813.15, 8813.2]], [[8813., 8814.]], .005,
                                   Path(folder)/"data")
            self.assertEqual(data.counts[30, 0], 1)
            self.assertEqual(data.counts[40, 0], 1)

    @unittest.skipUnless(importlib.util.find_spec("pynapple"), "optional Pynapple")
    def test_pynapple_adapter(self):
        import pynapple as nap
        epochs = nap.IntervalSet(start=[0., 2.], end=[1., 3.])
        units = nap.TsGroup({7: nap.Ts(t=[.2, 2.5]), 9: nap.Ts(t=[.1, .9])}, time_support=epochs)
        with tempfile.TemporaryDirectory() as folder:
            data = from_pynapple(units, epochs, .1, Path(folder)/"data")
            self.assertEqual(data.unit_ids, [7, 9])
            np.testing.assert_array_equal(data.counts.sum(axis=0), [2, 2])

    @unittest.skipUnless(importlib.util.find_spec("pynwb"), "optional NWB")
    def test_nwb_units_adapter(self):
        from pynwb import NWBFile, NWBHDF5IO
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"test.nwb"
            nwb = NWBFile("test fixture", "test", datetime.now(timezone.utc))
            nwb.add_unit(id=7, spike_times=[.1, .3, 2.7])
            nwb.add_unit(id=9, spike_times=[.4])
            with NWBHDF5IO(str(path), "w") as io:
                io.write(nwb)
            data = from_nwb(path, [[0, 1], [2, 3]], .1, Path(folder)/"data", unit_ids=[9, 7])
            self.assertEqual(data.unit_ids, [9, 7])
            np.testing.assert_array_equal(data.counts.sum(axis=0), [1, 3])

    def test_hnn_npz_extracts_without_losing_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"cache.npz"
            np.savez_compressed(path, spikes=self.y, gids=[3, 6, 8], bin_s=.001, segment_lengths=[50, 70])
            data = from_hnn_cache(path, Path(folder)/"converted")
            self.assertEqual(data.epoch_lengths, (50, 70))
            np.testing.assert_array_equal(data.counts, self.y)

    def test_adam_resume_matches_uninterrupted_and_rejects_wrong_data(self):
        model = StreamingGLM(self.basis, 32, rank=1, self_history=True)
        data = Recording(self.y)
        initial = model.initialize(3)
        continuous = fit_adam(model, data, initial, epochs=4, tolerance=-1)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/"checkpoint.npz"
            fit_adam(model, data, initial, epochs=2, tolerance=-1, checkpoint=checkpoint)
            resumed = fit_adam(model, data, epochs=2, tolerance=-1, checkpoint=checkpoint, resume=True)
            for key in initial:
                np.testing.assert_array_equal(continuous.params[key], resumed.params[key])
            other = self.y.copy(); other[0, 0] += 1
            with self.assertRaises(ValueError):
                fit_adam(model, Recording(other), checkpoint=checkpoint, resume=True)

    def test_save_load_prediction_identity(self):
        model = StreamingGLM(self.basis, 17, rank=1, self_history=True)
        params = model.initialize(3)
        data = Recording(self.y)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.npz"
            save_bundle(path, model, params, {"unit_ids": [0, 1, 2]})
            restored, p, meta, state = load_bundle(path)
            self.assertEqual(score(model, params, data), score(restored, p, data))
            self.assertEqual(meta["unit_ids"], [0, 1, 2])

    def test_cli_fit_resume_predict_evaluate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root/"data"
            from_spike_times([np.arange(.05, 8, .2), np.arange(.1, 8, .3)],
                             [[0, 8]], .1, data, unit_ids=[4, 9])
            out = root/"fit"
            common = ["fit", str(data), "--out", str(out), "--solver", "adam",
                      "--rank", "1", "--history", "3", "--n-basis", "2", "--chunk", "16", "--epochs", "1"]
            cli(common)
            cli(common+["--resume"])
            cli(["predict", str(out/"model.npz"), str(data), "--out", str(root/"prediction.npy")])
            pred = np.load(root/"prediction.npy")
            self.assertEqual(pred.shape, (80, 2))
            self.assertTrue(np.isnan(pred[:3]).all())
            self.assertTrue(np.isfinite(pred[3:]).all())
            cli(["evaluate", str(out/"model.npz"), str(data), "--out", str(root/"score.json")])
            self.assertTrue((root/"score.json").exists())

    def test_lbfgs_checkpoint_restart(self):
        model = StreamingGLM(self.basis, 32, rank=0, self_history=True, ridge=.1)
        data = Recording(self.y)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"checkpoint.npz"
            first = model.fit(data, max_iter=3, checkpoint=path)
            second = model.fit(data, max_iter=60, checkpoint=path, resume=True)
            self.assertTrue(second.converged)
            self.assertLessEqual(second.objective, first.objective)

    def test_float32_bundle_preserves_dtype_under_x64(self):
        model = StreamingGLM(self.basis, rank=1, dtype="float32")
        p = model.initialize(3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"f32.npz"
            save_bundle(path, model, p)
            restored, params, _, _ = load_bundle(path)
            self.assertEqual(str(restored.basis.dtype), "float32")
            self.assertEqual(str(params["b"].dtype), "float32")
            restored.value_and_grad(params, Recording(self.y))


if __name__ == "__main__":
    unittest.main()
