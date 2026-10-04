import unittest
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
from streamglm.tuned_comparison import select_candidate
from streamglm.model import StreamingGLM
from streamglm.data import Recording
from streamglm.reference import materialize_weights


class TunedSelectionTests(unittest.TestCase):
    def test_uses_validation_and_excludes_failed_gates(self):
        rows=[dict(status='DONE',gradient_gate_passed=True,validation={'log_likelihood_per_bin':-2},test={'gain':99}),
              dict(status='DONE',gradient_gate_passed=True,validation={'log_likelihood_per_bin':-1},test={'gain':-99}),
              dict(status='DONE',gradient_gate_passed=False,validation={'log_likelihood_per_bin':5}),
              dict(status='FAILED',gradient_gate_passed=True,validation={'log_likelihood_per_bin':9}),
              dict(status='DONE',gradient_gate_passed=True,validation={'log_likelihood_per_bin':float('nan')})]
        self.assertIs(select_candidate(rows),rows[1])
        self.assertIsNone(select_candidate(rows[2:]))

    def test_rank_penalty_matches_dense_weight_penalty(self):
        rng=np.random.default_rng(81)
        basis=np.array([[.7,.4],[.3,.6]])
        record=Recording(rng.poisson(.3,(35,5)))
        factored=StreamingGLM(basis,chunk_size=11,rank=2,ridge=.1)
        params=factored.initialize(5,seed=31)
        dense=StreamingGLM(basis,chunk_size=11,ridge=.1)
        full={'W':materialize_weights(params),'b':params['b']}
        actual,_=factored.value_and_grad(params,record)
        expected,_=dense.value_and_grad(full,record)
        self.assertAlmostEqual(actual,expected,places=11)
        unpenalized=StreamingGLM(basis,chunk_size=11)
        base,_=unpenalized.value_and_grad(full,record)
        self.assertAlmostEqual(expected-base,.05*np.sum(full['W']**2),places=11)

if __name__=='__main__': unittest.main()
