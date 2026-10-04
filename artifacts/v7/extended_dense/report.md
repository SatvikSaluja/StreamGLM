# Extended deterministic solver comparison

Fixed configuration; two seeds and two executions per worker. All measurements include the timing limitations in protocol.json.

| Neurons | Seed | Family | Backend | First s | Repeat s | Both pass gradient gate | Peak GiB |
|---|---|---|---|---:|---:|---|---:|
| 64 | 31 | lowrank | streamglm | 1.747 | 0.903 | True | 0.303 |
| 64 | 31 | lowrank | nemos | 7.901 | 2.176 | True | 0.639 |
| 64 | 31 | permutation_fullrank | streamglm | 4.251 | 3.716 | True | 0.306 |
| 64 | 31 | permutation_fullrank | nemos | 8.509 | 3.968 | True | 0.643 |
| 64 | 47 | lowrank | streamglm | 1.533 | 0.957 | True | 0.309 |
| 64 | 47 | lowrank | nemos | 6.653 | 2.093 | True | 0.647 |
| 64 | 47 | permutation_fullrank | streamglm | 4.877 | 4.146 | True | 0.312 |
| 64 | 47 | permutation_fullrank | nemos | 8.255 | 3.683 | True | 0.642 |
| 128 | 31 | lowrank | streamglm | 3.018 | 2.373 | True | 0.335 |
| 128 | 31 | lowrank | nemos | 8.764 | 3.989 | True | 0.710 |
| 128 | 31 | permutation_fullrank | streamglm | 10.554 | 10.057 | True | 0.340 |
| 128 | 31 | permutation_fullrank | nemos | 19.117 | 14.515 | True | 0.743 |
| 128 | 47 | lowrank | streamglm | 2.409 | 1.853 | True | 0.323 |
| 128 | 47 | lowrank | nemos | 8.723 | 4.031 | True | 0.724 |
| 128 | 47 | permutation_fullrank | streamglm | 13.017 | 12.498 | True | 0.338 |
| 128 | 47 | permutation_fullrank | nemos | 18.139 | 13.381 | True | 0.739 |
