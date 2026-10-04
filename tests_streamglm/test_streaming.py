import tempfile
from pathlib import Path
import unittest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from streamglm import Recording, StreamingGLM
from streamglm.reference import features, materialize_weights


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(31)
        self.y = self.rng.poisson(.2, (53, 4))
        self.basis = self.rng.normal(0, .2, (5, 3))
        self.lengths = [19, 3, 31]
        self.data = Recording(self.y, self.lengths)

    def test_objective_gradient_and_predictions_match_reference(self):
        x, valid = features(self.y, self.basis, self.lengths)
        for rank in (None, 2):
            for chunk in (2, 7, 128):
                with self.subTest(rank=rank, chunk=chunk):
                    model = StreamingGLM(self.basis, chunk, rank, ridge=.03)
                    p = model.initialize(4)
                    if rank is None:
                        p["W"] = jnp.asarray(self.rng.normal(0, .1, (3, 4, 4)))
                    def objective(q):
                        w = q["W"] if rank is None else jnp.einsum("kir,kjr->kij", q["U"], q["V"])
                        eta = jnp.einsum("tki,kij->tj", jnp.asarray(x[valid]), w) + q["b"]
                        return jnp.sum(jnp.exp(eta) - self.y[valid] * eta) / valid.sum() + .015 * jnp.sum(w*w)
                    expected, grad = jax.value_and_grad(objective)(p)
                    actual, streamed = model.value_and_grad(p, self.data)
                    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
                    for key in grad:
                        np.testing.assert_allclose(streamed[key], grad[key], rtol=1e-10, atol=1e-11)
                    predicted = np.concatenate([rates for _, rates, _ in model.predict_chunks(p, self.data)])
                    reference = np.exp(np.einsum("tki,kij->tj", x, materialize_weights(p)) + p["b"])
                    np.testing.assert_allclose(predicted[valid], reference[valid], rtol=1e-11)
                    self.assertTrue(np.isnan(predicted[~valid]).all())

    def test_causality_and_epoch_reset(self):
        y = np.zeros((12, 1), dtype=int)
        y[5] = 10
        model = StreamingGLM(np.ones((2, 1)), chunk_size=3)
        p = {"W": jnp.ones((1, 1, 1)), "b": jnp.zeros(1)}
        pred = np.concatenate([a for _, a, _ in model.predict_chunks(p, Recording(y, [6, 6]))])
        self.assertEqual(pred[5, 0], 1.)
        self.assertEqual(pred[8, 0], 1.)
        self.assertTrue(np.isnan(pred[6:8]).all())

    def test_disk_backed_recording(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "counts.npy"
            np.save(path, self.y)
            data = Recording.from_npy(path, self.lengths)
            self.assertIsInstance(data.counts, np.memmap)
            model = StreamingGLM(self.basis, chunk_size=7, rank=2)
            p = model.initialize(4)
            self.assertEqual(model.value_and_grad(p, data)[0], model.value_and_grad(p, self.data)[0])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Recording(self.y, [10, 10])
        bad = self.y.astype(float); bad[20, 0] = np.nan
        with self.assertRaises(ValueError):
            list(Recording(bad).batches(7, 5))
        model = StreamingGLM(self.basis, rank=2)
        with self.assertRaises(ValueError):
            model.value_and_grad(model.initialize(4), Recording(self.y[:3]))

    def test_fit_improves_objective(self):
        model = StreamingGLM(self.basis, chunk_size=32, ridge=.1)
        p = model.initialize(4)
        initial, _ = model.value_and_grad(p, self.data)
        fitted = model.fit(self.data, p, max_iter=60)
        self.assertTrue(fitted.converged, fitted.message)
        self.assertLess(fitted.objective, initial - .1)


if __name__ == "__main__":
    unittest.main()
