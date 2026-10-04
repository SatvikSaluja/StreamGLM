from __future__ import annotations
import argparse, csv, json, math
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from scipy.stats import spearmanr

def _coefficient_of_variation(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if math.isclose(mean, 0.0, abs_tol=1e-30):
        return float("nan")
    return float(np.std(arr, ddof=0) / abs(mean))

def _fmt_num(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return "nan"
        if abs(value) >= 1e4 or (0 < abs(value) < 1e-4):
            return f"{value:.4e}"
        return f"{value:.6g}"
    return str(value)

def _gid_position_map(net: Any) -> Dict[int, Tuple[float, float, float]]:
    gid_to_pos: Dict[int, Tuple[float, float, float]] = {}
    for cell_type in net.cell_types:
        gids = list(net.gid_ranges[cell_type]); positions = list(net.pos_dict[cell_type])
        if len(gids) != len(positions):
            raise RuntimeError(f"GID/position mismatch for {cell_type}")
        for gid, pos in zip(gids, positions):
            gid_to_pos[int(gid)] = tuple(float(v) for v in pos)
    return gid_to_pos

def _xy_distance(src_pos, target_pos) -> float:
    dx = float(target_pos[0]) - float(src_pos[0]); dy = float(target_pos[1]) - float(src_pos[1])
    return math.hypot(dx, dy)

def _source_sign(net: Any, src_type: str) -> str:
    electro_type = net.cell_types[src_type]["cell_metadata"].get("electro_type", "unknown")
    if electro_type == "excitatory": return "E"
    if electro_type == "inhibitory": return "I"
    return str(electro_type)

def _potential_pair_count(conn: Dict[str, Any]) -> int:
    src_gids = [int(gid) for gid in conn["src_gids"]]
    target_gids = [int(gid) for gid in conn["target_gids"]]
    n_possible = len(src_gids) * len(target_gids)
    if not bool(conn.get("allow_autapses", True)):
        n_possible -= len(set(src_gids).intersection(target_gids))
    return int(n_possible)

def _check_distance_scaling_invariants(weights, delays, *, A_weight, gain, A_delay,
                                        conn_idx, rtol=1e-12, atol=1e-15) -> None:
    expected_product = A_weight * gain * A_delay
    products = weights * delays
    if not np.allclose(products, expected_product, rtol=rtol, atol=atol):
        raise AssertionError(f"Connection {conn_idx}: weight*delay invariant failed.")
    if (len(weights) > 1 and np.unique(weights).size > 1 and np.unique(delays).size > 1
            and np.all(weights > 0) and np.all(delays > 0)):
        rho = float(spearmanr(weights, delays).statistic)
        if not math.isclose(rho, -1.0, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(
                f"Connection {conn_idx}: expected Spearman rho(weight, delay) == -1, got {rho:.16g}.")

def audit_network(net, coupling_window_ms: float = 25.0,
                  include_external_drives: bool = False):
    if coupling_window_ms <= 0: raise ValueError("coupling_window_ms must be > 0")
    inplane_distance = float(net._inplane_distance)
    if inplane_distance <= 0: raise ValueError("net._inplane_distance must be > 0")
    gid_to_pos = _gid_position_map(net); intrinsic_types = set(net.cell_types)
    summary_rows: List[Dict[str, Any]] = []; pair_rows: List[Dict[str, Any]] = []
    for conn_idx, conn in enumerate(net.connectivity):
        src_type = conn["src_type"]; target_type = conn["target_type"]
        if not include_external_drives and src_type not in intrinsic_types: continue
        if src_type not in intrinsic_types or target_type not in intrinsic_types: continue
        nc_dict = conn["nc_dict"]
        A_weight = float(nc_dict["A_weight"]); A_delay = float(nc_dict["A_delay"])
        lamtha = float(nc_dict["lamtha"]); gain = float(nc_dict.get("gain", 1.0))
        if lamtha <= 0: raise ValueError(f"Connection {conn_idx}: lamtha must be > 0")
        scaled_lamtha = lamtha * inplane_distance
        distances: List[float] = []; weights: List[float] = []; delays: List[float] = []
        for src_gid, target_gids in conn["gid_pairs"].items():
            src_gid = int(src_gid)
            if src_gid not in gid_to_pos: continue
            src_pos = gid_to_pos[src_gid]
            for target_gid in target_gids:
                target_gid = int(target_gid)
                if target_gid not in gid_to_pos: continue
                target_pos = gid_to_pos[target_gid]
                distance = _xy_distance(src_pos, target_pos)
                gaussian_factor = math.exp(-((distance / scaled_lamtha) ** 2))
                effective_weight = A_weight * gain * gaussian_factor
                effective_delay = (A_delay / gaussian_factor if gaussian_factor > 0 else float("inf"))
                distances.append(distance); weights.append(effective_weight); delays.append(effective_delay)
                pair_rows.append({"conn_idx": conn_idx, "src_type": src_type,
                    "target_type": target_type, "source_sign": _source_sign(net, src_type),
                    "receptor": conn["receptor"], "loc": conn["loc"], "src_gid": src_gid,
                    "target_gid": target_gid, "distance_xy_um": distance,
                    "inplane_distance_um": inplane_distance, "lamtha": lamtha,
                    "scaled_lamtha_um": scaled_lamtha, "A_weight": A_weight, "gain": gain,
                    "A_delay_ms": A_delay, "gaussian_factor": gaussian_factor,
                    "effective_weight": effective_weight, "effective_delay_ms": effective_delay,
                    "inside_coupling_window": (effective_delay <= coupling_window_ms)})
        if not weights: continue
        distances_arr = np.asarray(distances, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)
        delays_arr = np.asarray(delays, dtype=float)
        _check_distance_scaling_invariants(weights_arr, delays_arr, A_weight=A_weight,
            gain=gain, A_delay=A_delay, conn_idx=conn_idx)
        n_pairs = int(weights_arr.size); n_possible_pairs = _potential_pair_count(conn)
        realized_edge_fraction = (n_pairs / n_possible_pairs if n_possible_pairs > 0 else float("nan"))
        delay_min = float(np.min(delays_arr)); delay_max = float(np.max(delays_arr))
        rho_weight_delay = float(spearmanr(weights_arr, delays_arr).statistic)
        inside = delays_arr <= coupling_window_ms
        weight_min = float(np.min(weights_arr)); weight_max = float(np.max(weights_arr))
        summary_rows.append({"conn_idx": conn_idx,
            "class": f"{src_type}->{target_type}|{conn['receptor']}|{conn['loc']}",
            "src_type": src_type, "target_type": target_type,
            "source_sign": _source_sign(net, src_type), "receptor": conn["receptor"],
            "loc": conn["loc"], "probability": float(conn.get("probability", 1.0)),
            "allow_autapses": bool(conn.get("allow_autapses", True)), "n_pairs": n_pairs,
            "n_possible_pairs": n_possible_pairs, "realized_edge_fraction": realized_edge_fraction,
            "inplane_distance_um": inplane_distance, "lamtha": lamtha,
            "scaled_lamtha_um": scaled_lamtha, "A_weight": A_weight, "gain": gain,
            "A_delay_ms": A_delay, "distance_min_um": float(np.min(distances_arr)),
            "distance_max_um": float(np.max(distances_arr)), "weight_min": weight_min,
            "weight_max": weight_max,
            "weight_range_fraction_of_max": ((weight_max - weight_min) / weight_max if weight_max != 0 else float("nan")),
            "weight_cv": _coefficient_of_variation(weights_arr),
            "n_unique_weights": int(np.unique(weights_arr).size), "delay_min_ms": delay_min,
            "delay_max_ms": delay_max, "delay_span_ms": delay_max - delay_min,
            "n_unique_delays": int(np.unique(delays_arr).size),
            "n_pairs_inside_window": int(np.sum(inside)),
            "fraction_pairs_inside_window": float(np.mean(inside)),
            "weight_delay_spearman": rho_weight_delay})
    return summary_rows, pair_rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: List[Dict[str, Any]], window_ms: float) -> None:
    if not rows:
        print("No intrinsic connection specifications found.")
        return
    print(f"\nGROUND-TRUTH IDENTIFIABILITY AUDIT   (GLM coupling window 0-{window_ms:g} ms)")
    header = (f"{'idx':>3}  {'connection class':<46}{'N':>6}{'lam':>6}{'w_CV':>9}"
              f"{'uniqW':>7}{'delay range ms':>26}{'in_win':>8}{'rho':>7}")
    print(header); print("-" * len(header))
    for r in rows:
        print(f"{r['conn_idx']:>3}  {r['class']:<46.46}{r['n_pairs']:>6}{r['lamtha']:>6.0f}"
              f"{r['weight_cv']:>9.4f}{r['n_unique_weights']:>7}"
              f"{r['delay_min_ms']:>11.4g}..{r['delay_max_ms']:<13.4g}"
              f"{r['fraction_pairs_inside_window']:>7.1%}{r['weight_delay_spearman']:>7.1f}")
    print("\nInspect weight_cv, n_unique_weights, delay_span_ms and "
          "fraction_pairs_inside_window before choosing a headline metric.")


def main() -> None:
    p = argparse.ArgumentParser(description="Audit HNN-Core ground truth without simulating.")
    p.add_argument("--mesh", nargs=2, type=int, metavar=("NX", "NY"), default=(5, 5))
    p.add_argument("--window-ms", type=float, default=25.0)
    p.add_argument("--out-dir", type=Path, default=Path("audit_output"))
    p.add_argument("--no-pairs", action="store_true")
    args = p.parse_args()

    if min(args.mesh) < 1:
        raise SystemExit("--mesh values must be positive")
    if args.window_ms <= 0:
        raise SystemExit("--window-ms must be > 0")

    try:
        from hnn_core import neymotin_2020_model
    except ImportError as exc:
        raise SystemExit(
            "hnn_core is not installed. Install the version you want to audit:\n"
            "    pip install 'hnn-core==0.6.1'"
        ) from exc

    net = neymotin_2020_model(mesh_shape=tuple(args.mesh), legacy_mode=False)
    n_cells = sum(len(net.gid_ranges[t]) for t in net.cell_types)
    summary, pairs = audit_network(net, coupling_window_ms=args.window_ms)

    tag = f"{args.mesh[0]}x{args.mesh[1]}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / f"ground_truth_audit_{tag}.csv", summary)
    if not args.no_pairs:
        _write_csv(args.out_dir / f"ground_truth_pairs_{tag}.csv", pairs)

    try:
        version = metadata.version("hnn-core")
    except metadata.PackageNotFoundError:
        version = "unknown"

    (args.out_dir / f"ground_truth_audit_{tag}.json").write_text(json.dumps({
        "hnn_core_version": version,
        "mesh_shape": list(args.mesh),
        "n_real_cells": n_cells,
        "inplane_distance_um": float(net._inplane_distance),
        "coupling_window_ms": float(args.window_ms),
        "n_connection_specs": len(summary),
        "n_realized_pairs": len(pairs),
        "formula": {
            "distance": "xy plane only",
            "scaled_lamtha": "lamtha * inplane_distance",
            "gaussian_factor": "exp(-(distance_xy / scaled_lamtha)^2)",
            "effective_weight": "A_weight * gain * gaussian_factor",
            "effective_delay_ms": "A_delay / gaussian_factor",
            "within_class_invariant": "weight * delay == A_weight * gain * A_delay",
        },
    }, indent=2), encoding="utf-8")

    print(f"hnn-core {version} | mesh={tuple(args.mesh)} | cells={n_cells} | "
          f"inplane_distance={net._inplane_distance:g} um")
    _print_summary(summary, args.window_ms)
    print(f"\nwrote {args.out_dir}/")


if __name__ == "__main__":
    main()
