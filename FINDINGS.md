# Findings — Shadow Feature Selection Under Class Imbalance

## Headline

When the minority class is small, the shadow-feature heuristic on normalized
gain importance prunes genuinely predictive features at a rate that depends
**primarily on the absolute minority count `n_min = round(n·p)`**, not on
the imbalance ratio `p` itself. A credit-risk regime (p ∈ [0.05, 0.20], n
up to 5 000) and a fraud regime (p ∈ [0.001, 0.01], n up to 200 000) share
a common range of `n_min` (~10–2 000) and largely trace the same FNR-vs-`n_min`
curve.

See **`figures/01_collapse_fnr_max_delta_0.5.png`** for the headline plot
and **`figures/03_mechanism_decomposition.png`** for the mechanism.

## Representative numbers (δ = 0.5, n_noise = 5, XGBoost)

Per the full sweep (153 scenarios, 25 repeats below n = 50 000 and 15 above):

| ≈ n_min | scenario (regime / n / p) | mean noise floor | FNR_max | FNR_p90 |
|---:|:--|---:|---:|---:|
| 10 | B / 10 000 / 0.001 | 0.135 | **0.82** | 0.74 |
| 25 | A / 500 / 0.05 | 0.107 | **0.42** | 0.36 |
| 50 | A / 500 / 0.10 | 0.094 | 0.15 | 0.12 |
| 50 | A / 1 000 / 0.05 | 0.104 | 0.34 | 0.23 |
| 50 | B / 10 000 / 0.005 | 0.107 | 0.53 | 0.41 |
| 50 | B / 50 000 / 0.001 | 0.107 | 0.51 | 0.43 |
| 100 | A / 1 000 / 0.10 | 0.093 | 0.06 | 0.03 |
| 100 | B / 10 000 / 0.01 | 0.102 | 0.26 | 0.18 |
| 200 | A / 1 000 / 0.20 | 0.089 | 0.00 | 0.00 |
| 200 | B / 200 000 / 0.001 | 0.103 | **0.48** | 0.40 |
| 250 | A / 2 500 / 0.10 | 0.090 | 0.00 | 0.00 |
| 250 | B / 50 000 / 0.005 | 0.100 | 0.16 | 0.09 |
| 500 | A / 2 500 / 0.20 | 0.086 | 0.00 | 0.00 |
| 500 | B / 50 000 / 0.01 | 0.094 | 0.00 | 0.00 |
| 1 000 | A / 5 000 / 0.20 | 0.082 | 0.00 | 0.00 |
| 1 000 | B / 200 000 / 0.005 | 0.095 | 0.00 | 0.00 |
| 2 000 | B / 200 000 / 0.01 | 0.086 | 0.00 | 0.00 |

(Mean real-feature importance across all signal scenarios sits in
[0.103, 0.122]; at `n_min = 10` it's 0.106, **below** the mean noise floor
of 0.135. That inversion is the failure mode in one number.)

## Mechanism (figure 03)

The normalized-importance constraint (importances sum to 1 across features)
makes the noise share a function of the real features' absolute gain. As
`n_min` shrinks the trees' real-feature gain collapses toward the
noise-feature gain — not because the noise features find any signal, but
because the real ones can no longer find theirs.

The **pure-noise control** confirms this is a normalization effect rather
than absolute noise inflation: with no real signal, the normalized noise
floor sits at ≈ 0.10–0.13 across the entire `n_min` range and changes only
modestly (∼1.3× over a 200× change in `n_min`). With signal, the noise
floor starts at the same ≈ 0.13 at `n_min = 10` but falls to ≈ 0.08 at
`n_min = 2 000` — the steeper decline is the real features reclaiming
their share as the minority grows.

## Mitigation (figure 04)

Using the **90th-percentile** noise importance (a multi-shadow vote)
instead of the **maximum** typically reduces FNR by 5–15 absolute points at
intermediate `n_min` (e.g. 50–250) but does not save the heuristic at very
small `n_min`: at `n_min ≈ 10–25` the FNR_p90 is still 0.36–0.74.

## Caveats / residual structure

The collapse is strong but not perfect: at intermediate `n_min` (≈ 50–200)
the fraud regime (Regime B, p ≤ 0.005) sits roughly 0.1–0.2 FNR_max above
the credit-risk regime (Regime A) at the same `n_min`. Two of the most
extreme fraud cells (p = 0.001 at n ∈ {50 000, 200 000}) stay around
FNR ≈ 0.48–0.51 at `n_min` of 50 and 200, where Regime A collapses to ≈ 0
by `n_min ≈ 100`. So `n_min` is the dominant axis, but the very extreme
majority counts in the fraud regime carry a small residual penalty not
captured by `n_min` alone.

## Robustness (figures 07a, 07b)

- `n_noise = 20` shows the same shape and qualitative magnitudes, ruling
  out a 5-noise-feature artifact.
- The scikit-learn `GradientBoostingClassifier` cross-check reproduces the
  effect on Regime-A scenarios at δ = 0.5, confirming the result is not
  specific to XGBoost.

## Practical guideline

See `figures/summary_table.csv` (the paper's "Practical Guidelines" table).
Briefly:

| n_min bin | recommendation |
|---|---|
| `< 25` | shadow heuristic is unreliable; prefer stratified CV with direct gain ranking |
| `25–74` | unreliable; same recommendation |
| `75–199` | use the P90 floor (multi-shadow) rather than the max |
| `200–499` | use the P90 floor |
| `500–1499` | naive shadow heuristic OK |
| `1500+` | naive shadow heuristic OK |

The thresholds are aggressive only because the experiment uses a clean
synthetic Gaussian setup; real-world data with weaker signal will shift the
recommendations rightward.
