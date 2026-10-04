# Model and numerical conventions

For observed count vector y(t), temporal basis b(h,k), h=1..H:

    eta_j(t) = intercept_j + sum_{k,i,h} y_i(t-h) b(h,k) W(k,i,j)
    E[y_j(t) | observed history] = exp(eta_j(t))

Predictions are conditional expected counts per bin, not firing rates and not
free-running recurrent simulations. Bin duration is part of recording metadata.
Every evaluation requires a complete within-epoch history. Initial H bins of
each epoch or split are excluded, rather than padded with assumed zero history.
Chunk boundaries retain real overlap. Final partial chunks are padded only for
fixed JAX shapes, with an explicit loss mask.

Dense mode stores W with axes (basis, source, target). Factorized mode stores
U and V with axes (basis, neuron, rank), W_k=U_k V_k.T. Projection commutes with
shared linear temporal filtering, so the factorized path projects raw counts
before convolution. All chunks use the same current parameter values during
an exact full-data objective/gradient pass.

With `self_history=True`, the diagonal of W is removed and replaced by an
independent coefficient S(k,j). Rank zero plus self-history is the independent
self-history baseline. Off-diagonal truncation plus a free diagonal need not
itself be a rank-r matrix; the term 'rank' refers to the underlying factors.

The objective is:

    mean_t sum_j [exp(eta_j(t)) - y_j(t)*eta_j(t)] + ridge/2 * ||effective_W||²

Log-factorials are omitted only during optimization. Evaluation includes them.
Intercepts are unpenalized. The low-rank penalty uses Gram matrices, subtracting
the coupling diagonal and adding S when applicable. It never constructs an
N-by-N matrix during fitting. Export/reference routines can materialize it
explicitly on small data.

No clipping modifies the exp link. Nonfinite objective/gradient or predictions
raise errors. Large rates or poorly scaled parameters require rescaling,
regularization, or a smaller Adam learning rate, rather than silently changing
the fitted likelihood. All CLI commands use float64. The library has an explicit
dtype option and never toggles global JAX precision on import.

## Optimizers and memory

L-BFGS is full-data optimization with streamed objective evaluation. Its line
search can require multiple passes per iteration. Optimizer history scales with
parameter count, not recording duration. Each chunk's reverse pass finishes
before advancing; JAX checkpointing recomputes basis intermediates. Framework
and compiler workspace must still be measured.

Adam visits sequential chunks and updates parameters once per valid chunk.
It is deterministic, not a shuffled sampling scheme. Each completed epoch gets
an additional full-data objective/gradient check. Its stopping rule is a full
gradient infinity norm tolerance. Running out of epochs reports nonconvergence.
L-BFGS can stop by relative objective reduction; the report retains its actual
message and gradient norm rather than claiming that every convergence flag
means the same criterion.

Adam checkpoints include first/second moments and update count, and resume at
completed epoch boundaries. L-BFGS checkpoints contain accepted parameters;
restart reinitializes its history. Checkpoint fingerprints cover all counts,
unit IDs, epoch lengths and bin duration. Resume also checks basis, precision,
chunk length, regularization and optimizer configuration.

## Data conversion

NWB/Pynapple imports use explicit half-open epochs and read one unit's spikes at
a time. Output counts are int32 memory maps. A unit's spike train and its
per-epoch histogram can be resident during conversion; this is not claimed to
use only one fit-sized chunk. Tiny floating-point discrepancies at nominal bin
edges are snapped within a bound based on machine precision and timestamp size.
Incomplete final bins are rejected. Metadata includes the original interval
times, bin width, unit order and source identifier.

HNN import copies the compressed spikes NPY member in bounded byte blocks and
preserves explicit segment lengths. It does not rerun HNN or infer missing
segmentation. Data gaps must be separate epochs; filling them with zeros changes
the conditioning history and is not an acceptable substitute.

## Statistical interpretation

Model selection uses validation likelihood, with final test data held out.
Training-only mean rates define the intercept baseline and firing-rate filter.
Candidate fits share the same retained units and valid evaluation rows.
The self-history baseline controls for a neuron's own recent spikes.

Predictive coupling can reflect common inputs, stimulus or behavior. No
anatomical interpretation follows from likelihood improvement alone. Low-rank
factor axes can rotate or rescale while preserving W, so analysis should use
reconstructed interactions or invariant subspaces, not label individual factors
as uniquely identified brain mechanisms.
