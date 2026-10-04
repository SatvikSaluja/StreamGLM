import tempfile,unittest
from pathlib import Path
import numpy as np
from streamglm.coupled_scale import generate
from streamglm.reference import features,materialize_weights

class CoupledScaleTests(unittest.TestCase):
    def test_resume_is_exact_and_inhibitory_truth_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b'
            generate(a,12,100,12,stop_after=43);generate(a,12,100,12);generate(b,12,100,12)
            np.testing.assert_array_equal(np.load(a/'counts.npy'),np.load(b/'counts.npy'))
            with np.load(a/'truth.npz') as z:p={k:z[k] for k in z.files}
            w=materialize_weights(p);self.assertTrue((w<=0).all());self.assertTrue((w<0).any())
            # Replay RNG and direct materialized history, independent of latent generator.
            rng=np.random.default_rng(12)
            for _ in range(2):rng.uniform(.8,1.2,12);rng.uniform(.8,1.2,12)
            y=np.zeros((100,12),dtype=int);filters=np.einsum('hk,kij->hij',p['basis'],w)
            for t in range(100):
                h=min(t,8);eta=p['b'].copy()
                if h:eta+=np.einsum('hi,hij->j',y[t-h:t][::-1],filters[:h])
                self.assertTrue((np.exp(eta)<=.20000001).all());y[t]=rng.poisson(np.exp(eta))
            np.testing.assert_array_equal(y,np.load(a/'counts.npy'))
if __name__=='__main__':unittest.main()
