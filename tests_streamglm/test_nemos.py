"""Independent NeMoS convolution, likelihood, prediction and gradient oracle."""
import importlib.util
import inspect
import unittest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln
from streamglm import Recording, StreamingGLM
from streamglm.reference import materialize_weights


@unittest.skipUnless(importlib.util.find_spec("nemos"), "install streamglm[validation]")
class NemosTests(unittest.TestCase):
    def test_export_with_self_history(self):
        import nemos as nmo
        from streamglm.integration import to_nemos
        rng = np.random.default_rng(5)
        counts = rng.poisson(.2, (40, 4))
        basis = rng.normal(0, .1, (5, 2))
        model = StreamingGLM(basis, chunk_size=11, rank=2, self_history=True)
        params = model.initialize(4)
        params["S"] = jnp.ones((2, 4))*.2
        exported = to_nemos(model, params)
        x = nmo.convolve.create_convolutional_predictor(basis, counts, shift=True)
        expected = np.asarray(exported.predict(x[5:].reshape(35, 8)))
        actual = np.concatenate([p for _, p, _ in model.predict_chunks(params, Recording(counts))])[5:]
        np.testing.assert_allclose(actual, expected, atol=1e-11)
        with self.assertRaises(MemoryError):
            to_nemos(model, params, max_bytes=1)

    def test_nemos_fixed_parameter_equivalence(self):
        import nemos as nmo
        rng = np.random.default_rng(93)
        basis = rng.normal(0, .15, (7, 3))
        n = 20
        counts = rng.poisson(.2, (101, n))
        lengths = [43, 58]
        # NeMoS builds its own causal features independently for each epoch.
        x = np.concatenate([np.asarray(nmo.convolve.create_convolutional_predictor(
            basis, y, predictor_causality="causal", shift=True))
            for y in np.split(counts, [43])])
        valid = np.isfinite(x).all(axis=(1, 2))
        design = jnp.asarray(x[valid].reshape(valid.sum(), -1))
        for rank in (None, 2):
            with self.subTest(rank=rank):
                model = StreamingGLM(basis, chunk_size=11, rank=rank)
                p = model.initialize(n)
                if rank is None:
                    p["W"] = jnp.asarray(rng.normal(0, .1, (3, n, n)))
                w = materialize_weights(p)
                if "inverse_link_function" in inspect.signature(nmo.glm.PopulationGLM).parameters:
                    oracle = nmo.glm.PopulationGLM(observation_model="Poisson", inverse_link_function=jnp.exp)
                else:
                    oracle = nmo.glm.PopulationGLM(observation_model=
                        nmo.observation_models.PoissonObservations(inverse_link_function=jnp.exp))
                # Set identical parameters, rather than comparing different fitted models.
                oracle.coef_ = jnp.asarray(w.transpose(1, 0, 2).reshape(3*n, n))
                oracle.intercept_ = p["b"]
                oracle.scale_ = 1.
                predicted = np.concatenate([a for _, a, _ in model.predict_chunks(p, Recording(counts, lengths))])
                np.testing.assert_array_equal(np.isfinite(predicted).all(axis=1), valid)
                np.testing.assert_allclose(predicted[valid], oracle.predict(design), rtol=1e-11, atol=1e-12)
                value, gradient = model.value_and_grad(p, Recording(counts, lengths))
                score = oracle.score(design, counts[valid], score_type="log-likelihood",
                                     aggregate_sample_scores=lambda z: jnp.sum(jnp.mean(z, axis=0)))
                factorial = np.sum(gammaln(counts[valid] + 1)) / valid.sum()
                np.testing.assert_allclose(value + factorial, -score, rtol=1e-11, atol=1e-11)

                def loss(q):
                    weights = q["W"] if rank is None else jnp.einsum("kir,kjr->kij", q["U"], q["V"])
                    coef = weights.transpose(1, 0, 2).reshape(3*n, n)
                    rate = jnp.exp(design @ coef + q["b"])
                    return -oracle.observation_model.log_likelihood(
                        jnp.asarray(counts[valid]), rate,
                        aggregate_sample_scores=lambda z: jnp.sum(jnp.mean(z, axis=0)))

                reference_gradient = jax.grad(loss)(p)
                for key in p:
                    np.testing.assert_allclose(gradient[key], reference_gradient[key], rtol=1e-10, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
