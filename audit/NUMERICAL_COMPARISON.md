# Comparing reconstructed sequential summaries

The model-free audit reconstructs all 20 sequential summaries from the
retained arrays. Derived floating-point reductions use the quantity-specific
binary64 comparison policy below. The original arrays, result files and
summary-producing functions are retained unchanged.

From `audit/`, in an environment with `requirements-checks.txt` installed:

```sh
python -B -m unittest -v
python -B reconstruct_summaries.py --output /tmp/pldr-foundations-reconstruction.json
```

## Exact checks

The six raw/result files are SHA-256 pinned in `retained-inputs.json`.
Their bytes, including metadata and array shapes, must match. The comparison
also checks summary key sets, list lengths, integer counts and indices, and
Boolean decisions exactly. Booleans are checked before numeric coercion.
Unregistered fields fail even if both summaries contain them. No NaN or
infinity sentinel is admitted in sequential summaries. Count-derived accuracy
fractions, signs and exact-zero decisions are exact.

`sequential-decisions.json` binds the individual maximizer sets (including
ties), pair signs, half-margin gates and held-out per-token signs. Independent
recomputation uses `math.fsum` for candidate totals and compares serialized
decisions with exact types. This checks individual decisions in addition to
the summary counts.

## Floating-point budgets

For binary64 unit roundoff `u = 2^-53`, put `gamma(n) = n*u/(1-n*u)`.
The policy rejects reduction sizes with `n*u >= 0.01`. Let `T` be token count,
`I` item count, `L` maximum candidate token length, `K` maximum candidate
count, `M` the maximum absolute primitive log probability, and `A` the larger
mean absolute log probability of the two scoring protocols. Define:

- `e_token = 4*u*M`, allowing two subtraction paths;
- `e_candidate = 4*gamma(L+2)*L*M`, covering candidate reductions and subtraction;
- `e_mass = 2*e_candidate + 4*gamma(K+8)`, propagating score errors through
  probability mass, shifted exponentials, summation and division;
- `r = 2*gamma(max(T,I)+8)`, covering the final reduction/comparison paths.

The exponentiation allowance assumes at most four units of relative roundoff
per finite exponential in the supported stack. These are conservative operation
budgets, not platform-independent correctly-rounded-libm theorems. Retained
finite arrays are the domain. A new dtype, nonfinite policy, reduction formula
or unsupported library law requires requalification.

| Quantity class | Absolute allowance `a` | Relative allowance |
| --- | --- | --- |
| NLL/CE means | 0 | `r` |
| Difference of NLL/CE means | `4*gamma(T+2)*A` | `r` |
| Token-gap statistics and position medians | `e_token` | `r` |
| Candidate-gap and margin statistics | `e_candidate` | `r` |
| MC2 probability-mass means | `e_mass` | `r` |
| MC2 gap, signed-gap and absolute-gap statistics | `2*e_mass + 4*u` | `r` |

The comparator requires
`abs(actual-expected) <= a + r*max(abs(actual),abs(expected))`, together with
the exact checks above. In particular, the approximately `-8.45e-10` MC2
mean gap cannot be erased or change sign. Nine-significant-digit agreement
is reported as an additional diagnostic; it does not define the tolerance.
The reconstruction report records discrepancies and budgets by quantity class.

## Independent reference and rejection controls

The cancellation-sensitive MC2 means and signed medians also have an
80-digit Decimal reference, using exact conversion of the retained binary64
inputs, independent sums and Decimal exponentials. Differences from this
reference and differences between two binary64 reconstructions are reported
separately. The high-precision comparison is an additional numerical check;
it does not replace the archived measurements. Some cancellation-sensitive
diagnostics change their nine-digit rendering against the Decimal reference
within the declared budgets.

Negative tests cover corrupted input hashes and categorical metadata, missing
keys, altered shape, changed counts, Boolean/integer confusion, unknown fields,
nonfinite values, changed signs or zero decisions, ties, and one representable
step beyond each quantity class's comparison budget.

## Provenance and scope

The comparator, reconstruction script, rejection tests and individual-decision
manifest are adapted from the public
[Book Companion implementation](https://github.com/burcgokden/PLDR-LLM-Book-Companion/tree/dcd094a9c4f90ed3250c967852a48c4560759604/companions/foundations/audit)
and its
[numerical contract](https://github.com/burcgokden/PLDR-LLM-Book-Companion/blob/dcd094a9c4f90ed3250c967852a48c4560759604/docs/AUDIT.md).
All six retained files and both sequential/online helper modules are
byte-identical between that source and Foundations commit
`a87ccc1949e57d20640ec425ab24138401641aa6`. The input manifest keeps the same
six hashes with a Foundations schema label. This establishes that the
comparison policy applies to the same inputs and summary formulas.

The original sequential test compared serialized floating-point summaries for
exact string equality. On Python 3.14.6, both NumPy 2.4.4 and 2.5.2 reconstruct
the same 478 floating fields, with seven last-bit differences from the archive
and maximum absolute difference `2.2248379150818266e-18`. All nine-significant-digit
binary64 summaries and all 522 discrete fields agree. The NumPy version switch
alone does not explain the original failure; pinning NumPy alone does not fix it.
The remaining historical cause of those last-bit differences is undetermined.

This policy applies to offline sequential-summary reconstruction. The existing
main and online audit checks remain in place. Fixed-environment, repeated-run
determinism of the original model acquisition is a separate claim; this
reconstruction check neither reruns the models nor changes that claim.
