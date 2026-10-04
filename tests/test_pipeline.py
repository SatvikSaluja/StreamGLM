"""End-to-end checks. Runs in seconds, no NEURON required.

This is what CI runs: the synthetic path end to end plus the ground-truth
resolver against a mock that reproduces hnn-core v0.6.1 data structures.
Full HNN simulations do not run here (PROJECT.md section 26).

    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from mock_hnn import MockNet  # noqa: E402

from neurotwinbench import fit as fitmod  # noqa: E402
from neurotwinbench import metrics as met  # noqa: E402
from neurotwinbench import synthetic  # noqa: E402
from neurotwinbench.ground_truth import extract_ground_truth  # noqa: E402


def test_component_self_checks() -> None:
    synthetic._self_check()
    met._self_check()


def test_mesh_5x5_gives_70_cells() -> None:
    """Basket placement is a sub-lattice, not a full grid: 10 per layer at 5x5."""
    for mesh, expected in [((5, 5), 70), ((10, 10), 270)]:
        net = MockNet(mesh_shape=mesh)
        n = sum(len(net.gid_ranges[t]) for t in net.cell_types)
        assert n == expected, f"mesh {mesh}: got {n}, expected {expected}"
    print("ok: 5x5 -> 70 cells, 10x10 -> 270 cells")


def test_weight_delay_rank_confounded() -> None:
    """PROJECT.md section E: rho(weight, delay) == -1 in every connection class.

    If this ever fails, either hnn-core's scaling changed or the network has
    delays that vary independently of weight -- which is the condition section 15
    needs and would want to be told about.
    """
    gt = extract_ground_truth(MockNet(mesh_shape=(5, 5)))
    n = 0
    for name, mask in gt.by_class():
        rho = spearmanr(np.abs(gt.weight[mask]), gt.delay_ms[mask]).statistic
        assert np.isclose(rho, -1.0, atol=1e-9), f"{name}: rho={rho}"
        n += 1
    print(f"ok: rho(|w|,delay) == -1 in all {n} connection classes")


def test_lamtha_is_not_a_valid_grouping_key() -> None:
    """Pooling classes that share a lamtha but differ in A_weight is an artifact.

    Guards the mistake directly: at 5x5, lamtha=50 spans L2bask->L2pyr
    (A_weight 5e-2) and L2bask->L5pyr (1e-3). Per class each reads CV 0.0027
    and rho = -1; pooled they read CV 0.686 and rho = -0.554.
    """
    gt = extract_ground_truth(MockNet(mesh_shape=(5, 5)))
    pooled = gt.lamtha == 50.0
    rho_pooled = spearmanr(np.abs(gt.weight[pooled]), gt.delay_ms[pooled]).statistic
    assert rho_pooled > -0.99, (
        "lamtha=50 no longer pools distinct A_weight scales -- if hnn-core changed, "
        "re-check every per-class claim in PROJECT.md section F"
    )
    for name, mask in gt.by_class():
        if not (gt.lamtha[mask] == 50.0).all():
            continue
        rho = spearmanr(np.abs(gt.weight[mask]), gt.delay_ms[mask]).statistic
        assert np.isclose(rho, -1.0, atol=1e-9), f"{name}: rho={rho}"
    print(f"ok: pooled-by-lamtha rho={rho_pooled:+.3f}, per-class rho=-1.000")


def test_dynamic_range_splits_on_ei_axis() -> None:
    """Every pyramidal-source class has usable weight spread; basket-source does not."""
    gt = extract_ground_truth(MockNet(mesh_shape=(5, 5)))
    cv = {}
    for name, mask in gt.by_class():
        w = np.abs(gt.weight[mask])
        cv[name] = (float(gt.lamtha[mask][0]), float(np.std(w) / np.mean(w)))
    exc = [c for lam, c in cv.values() if lam == 3.0]
    inh = [c for lam, c in cv.values() if lam >= 50.0]
    assert min(exc) > 0.4, f"pyramidal-source spread collapsed: {cv}"
    assert max(inh) < 0.05, f"inhibitory spread unexpected: {cv}"
    print(f"ok: pyramidal-source CV {min(exc):.4f}-{max(exc):.4f}, "
          f"basket->pyramidal CV {min(inh):.4f}-{max(inh):.4f}")


def test_edge_detection_chance_levels() -> None:
    """A noise estimator must score at chance, and chance differs per metric.

    ROC-AUC has a fixed 0.5 baseline; PR-AUC's baseline IS the prevalence. A
    PR-AUC of 0.24 looks poor until you know 24% of candidate pairs are edges,
    at which point it is exactly chance -- which is why edge_detection returns
    prevalence alongside.
    """
    rng = np.random.default_rng(0)
    n = 40
    connected = rng.random((n, n)) < 0.25
    np.fill_diagonal(connected, False)

    noise = met.edge_detection(connected, rng.normal(0, 1, (n, n)))
    assert abs(noise["roc_auc"] - 0.5) < 0.1, noise
    assert abs(noise["pr_auc"] - noise["prevalence"]) < 0.1, noise

    signal = connected * rng.gamma(2, 0.5, (n, n)) + rng.normal(0, 0.3, (n, n))
    good = met.edge_detection(connected, signal)
    assert good["roc_auc"] > 0.75, good
    print(f"ok: noise roc={noise['roc_auc']:.3f} pr={noise['pr_auc']:.3f} "
          f"(prevalence {noise['prevalence']:.3f}) | signal roc={good['roc_auc']:.3f}")


def test_edge_detection_refuses_dense_networks() -> None:
    """With no designed non-edges this is not a detection task (section 2B)."""
    n = 20
    dense = np.ones((n, n), dtype=bool)
    np.fill_diagonal(dense, False)
    out = met.edge_detection(dense, np.random.default_rng(0).normal(size=(n, n)))
    assert np.isnan(out["roc_auc"]), "a dense network must not yield an AUC"
    assert out["n_non_edges"] == 0
    print("ok: dense network returns nan rather than a spurious AUC")


def test_silent_neurons_leave_the_evaluation_set() -> None:
    """A neuron that never fired has unrecoverable coupling, by any estimator."""
    rng = np.random.default_rng(0)
    spikes = rng.poisson(0.01, size=(20000, 8)).astype(float)
    spikes[:, 3] = 0.0
    spikes[:, 6] = 0.0

    result = fitmod.fit_population_glm(spikes, history_ms=10.0, n_basis=4)
    assert not result.active[3] and not result.active[6], result.active
    assert np.all(result.filters[3] == 0) and np.all(result.filters[:, 3] == 0)
    assert not result.evaluable[3].any() and not result.evaluable[:, 6].any()
    print(f"ok: 2 silent neurons dropped, {int(result.evaluable.sum())}/"
          f"{result.evaluable.size} pairs evaluable")


def test_sign_consistency_beats_sd_as_a_reliability_measure() -> None:
    """Which per-connection uncertainty measure actually discriminates.

    Fold-to-fold magnitude variability barely separates real connections from
    spurious ones; sign agreement separates them cleanly, and unconnected pairs
    land at chance. Pinned because reporting `sd` instead would look rigorous
    while carrying almost no information.
    """
    net = synthetic.simulate(n_neurons=8, duration_s=60.0, strength=0.15, seed=0)
    connected = net.filters.sum(axis=2) != 0.0

    st = fitmod.coupling_stability(net.spikes, n_folds=4, history_ms=15.0, n_basis=4)
    sign_gap = st["sign_consistency"][connected].mean() - \
               st["sign_consistency"][~connected].mean()
    sd_gap = abs(st["sd"][connected].mean() - st["sd"][~connected].mean())

    assert st["sign_consistency"][connected].mean() > 0.75, st["sign_consistency"]
    assert abs(st["sign_consistency"][~connected].mean() - 0.5) < 0.15, "should be chance"
    assert sign_gap > sd_gap, f"sign gap {sign_gap:.3f} should exceed sd gap {sd_gap:.3f}"
    print(f"ok: sign consistency {st['sign_consistency'][connected].mean():.2f} vs "
          f"{st['sign_consistency'][~connected].mean():.2f} (gap {sign_gap:.2f}); "
          f"sd gap only {sd_gap:.2f}")


def test_uninformative_drive_explains_nothing() -> None:
    """Section 4's measurement must return ~0 for a drive unrelated to the data."""
    net = synthetic.simulate(n_neurons=8, duration_s=60.0, strength=0.15, seed=0)
    sham = np.random.default_rng(0).poisson(0.02, size=(net.spikes.shape[0], 1))
    out = fitmod.drive_contribution(net.spikes, sham, history_ms=15.0, n_basis=4)
    assert abs(out["population_pseudo_r2"]) < 0.01, out["population_pseudo_r2"]
    print(f"ok: sham drive pseudo-R2 = {out['population_pseudo_r2']:+.5f}")


def test_estimator_beats_its_null() -> None:
    """Smoke test that the pipeline is wired correctly end to end.

    Thresholds are deliberately loose. Spearman over fewer than ~30 connected
    pairs is too noisy to separate signal from the null at all, so the network
    has to be big enough to make the metric itself stable -- 14 neurons gives
    ~45 connections. Recovery here still sits below what a full-length run
    reaches (see results/); that gap is the data-budget question of section 5,
    not a defect. Establishing the real calibration tolerance is
    experiments/calibrate.py's job. This test only has to fail loudly if the
    pipeline breaks.
    """
    net = synthetic.simulate(n_neurons=14, duration_s=150.0, strength=0.15, seed=0)
    true_area = net.filters.sum(axis=2)
    connected = true_area != 0.0

    result = fitmod.fit_population_glm(net.spikes, history_ms=25.0, n_basis=8)
    glm = met.evaluate(true_area, result.areas, connected)

    jittered = met.jitter(net.spikes, width_bins=50, seed=1)
    null = met.evaluate(true_area, fitmod.fit_population_glm(jittered).areas, connected)

    assert glm.weight_spearman > 0.2, f"recovery collapsed: {glm}"
    assert glm.sign_accuracy > 0.85, f"sign recovery collapsed: {glm}"
    assert glm.weight_spearman > 2 * abs(null.weight_spearman), (
        f"GLM does not clear its null: glm={glm} null={null}"
    )
    print(f"ok: GLM {glm}\n    null {null}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"\n--- {name} ---")
            fn()
    print("\nall checks passed")
