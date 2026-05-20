# Section 3.1.2 PaDiM Observation

This package reproduces PaDiM representation-level equivalent conditions:

- marginal subspace readouts;
- instability vs AUROC;
- instability vs error;
- risk-coverage curve.

## Outputs

- `outputs/figures/sec3_padim/`
- `outputs/cached_results/sec3_padim/`

### Paper-style Figure 2 (2×2, evidence layout)

**Diagnostic figures** (for example `padim_marginal_scatter.png` from Stage 2 plotting) are **validation-style** readouts. **Paper-style Figure 2** is a separate cached-only build:

```bash
python3 src/experiments/sec3_padim_observation/build_paper_style_fig2_padim.py
```

It reads `marginal_protocol_b.csv` (run-level **Fused AUROC** and **mean instability**) and `outputs/cached_results/raw_scores/padim/unified_raw_scores_wide.csv` to derive **strict pairwise** (three-view) statistics and **PromptAD-aligned** risk–coverage curves **without** re-running PaDiM or re-aggregating Protocol B raw evidence.

Outputs:

- `outputs/figures/sec3_padim/paper_style_fig2.png` / `.pdf`
- `outputs/cached_results/sec3_padim/paper_style_fig2_summary.json`

Panels **(a)–(c)** use **run-level** aggregates from the marginal table (and the same ranking-error complement as Sec 3.1.1: `1 - fused_AUROC`, labeled **mean ranking error**). Panel **(d)** uses a **different** seed pair than (a): among all same-(dataset, category, backbone) pairs with **|ΔAUROC| ≤ 0.005**, both seeds must have marginal AUROC/instability inside the **[p1, p99]** band over the cached grid (outliers dropped; filter can relax if the pool is empty). Candidates are scored by  
`high_coverage_gap + 0.5 * mean_risk_gap + 0.2 * normalized_instability_gap − penalty(auroc_gap)`  
(high-coverage gap = mean |Δrisk| at merged coverage **≥ 0.8**). The subplot title is **`Reliability vs coverage`** only when that best **high_coverage_gap ≥ 0.005**; otherwise it is **`Risk–coverage comparison`** and `paper_style_fig2_summary.json` sets **`promptad_vs_padim_risk_coverage`** — PaDiM risk–coverage separation for these near-AUROC pairs is often **weaker than on PromptAD Sec 3.1.1**, so avoid claiming “clearly different decision reliability” from panel (d) alone.

Requires **SciPy** (Spearman / optional KDE), same as the PromptAD paper-style builder.

## Run

Default (**fast path**): copies bundled CSV stubs; does **not** run PaDiM Protocol B.

```bash
bash src/experiments/sec3_padim_observation/run.sh
```

**Two-stage:** **Stage 1** — `FULL_RUN=1 bash scripts/run_padim_raw.sh` with `PADIM_*` (optional **`PADIM_PROFILE=debug|paper`**; optional **`PADIM_CLASSES`** overrides profile — see **`docs/FULLPATH_PADIM.md`**). **Stage 2** — `bash scripts/reproduce_sec3_padim.sh` (plots from `marginal_protocol_b.csv` or bundled stub; **never** runs Stage 1). See **`docs/FULLPATH_PADIM.md`**, **`docs/REPRODUCE.md`**.
