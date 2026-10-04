# Quick deterministic solver pilot

One seed; fixed ridge; two executions per worker. No held-out test selection. See protocol.json for timing boundaries.

| Dataset | Backend | First fit s | Repeated fit s | Gradient gate (both) | Peak GiB |
|---|---|---:|---:|---|---:|
| lowrank | streamglm | 1.443 | 0.786 | True | 0.303 |
| lowrank | nemos | 7.794 | 2.193 | True | 0.638 |
| permutation_fullrank | streamglm | 4.621 | 3.691 | True | 0.306 |
| permutation_fullrank | nemos | 8.507 | 4.162 | True | 0.654 |
