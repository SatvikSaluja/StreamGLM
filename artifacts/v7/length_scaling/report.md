# Recording-length scaling pilot

Fixed configuration; one seed and two executions per worker. All measurements include the timing limitations in protocol.json.

| Bins | Seed | Family | Backend | First s | Repeat s | Both pass gradient gate | Peak GiB |
|---|---|---|---|---:|---:|---|---:|
| 24000 | 31 | lowrank | streamglm | 2.379 | 1.606 | True | 0.311 |
| 24000 | 31 | lowrank | nemos | 7.863 | 2.781 | True | 0.730 |
| 24000 | 31 | permutation_fullrank | streamglm | 7.446 | 6.436 | True | 0.315 |
| 24000 | 31 | permutation_fullrank | nemos | 11.489 | 6.545 | True | 0.723 |
| 48000 | 31 | lowrank | streamglm | 3.960 | 3.274 | True | 0.316 |
| 48000 | 31 | lowrank | nemos | 8.413 | 3.627 | True | 0.814 |
| 48000 | 31 | permutation_fullrank | streamglm | 13.007 | 12.867 | True | 0.309 |
| 48000 | 31 | permutation_fullrank | nemos | 14.459 | 9.781 | True | 0.879 |
| 96000 | 31 | lowrank | streamglm | 5.605 | 4.802 | True | 0.335 |
| 96000 | 31 | lowrank | nemos | 9.809 | 5.034 | True | 1.018 |
| 96000 | 31 | permutation_fullrank | streamglm | 22.125 | 21.315 | True | 0.332 |
| 96000 | 31 | permutation_fullrank | nemos | 21.120 | 16.682 | True | 1.020 |
