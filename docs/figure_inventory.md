# Figure reproduction inventory

`learning_trajectory.figures.make_figures(output="outputs")` creates all six main figures and all ten supplementary figures as PNG and editable SVG. The pipeline requires completed numerical tables and raises an error when a required table or endpoint is missing. No published reference value is substituted for a computed observation.

Figures reproduce the scientific panels and analysis choices, with a newly typeset layout. They are not pixel copies of the published figures. `outputs/figures/manifest.json` records the input CSV fingerprints.

For a direct original/reconstruction comparison, run `.venv/bin/python scripts/build_figure_comparison.py`. This creates `figures/figure1_orig.png` alongside `figures/figure1_recon.png` (and the analogous names for 1–6 and S1–S10), with original captions retained. `figures/index.html` displays all 16 pairs side by side; reconstruction SVGs are included. The original PDF-region coordinates and hashes are recorded in `figures/original_manifest.json`, and all paired images in `figures/comparison_manifest.json`.

Pooled endpoints and complete-run deletion replicates used by Figures 3, S8 and S9 are also retained as `outputs/summary/pooled_estimates.csv` and `outputs/summary/pooled_jackknife_replicates.csv` (273 estimates and 8,190 replicates for the full paper design).

| Output basename | Reproduced panels | Computed sources |
| --- | --- | --- |
| `figure_01` | Cohort/matching design; final within/between canonical Alpha | Training inventory; direct-profile canonical Alpha estimates |
| `figure_02` | Canonical Alpha trajectory; layer contrasts; Euclidean/SND layer contrasts | Main-cohort estimates |
| `figure_03` | Pooled raw cosine; within/between canonical Alpha; activation correspondence for all three alignment conditions | Main pair tables; main estimates; pooled run jackknife |
| `figure_04` | Paired task design; task-contrast trajectories; final task contrasts | Task training inventory and paired-block estimates |
| `figure_05` | Raw W0 Alpha and CKA contrasts; endpoints; pairwise scatter and conditional correlations | Main estimates, pair tables, correlation estimates |
| `figure_06` | First 64 SGD updates: accuracy, relative displacement, canonical Alpha/CKA separation | Early training and displacement runs; early jackknife estimates |
| `figure_S01` | Individual/mean accuracy and loss; final run-level distributions | Main training runs |
| `figure_S02` | Three raw metrics × four layer/aggregation views, within and between trajectories | Main raw Alpha, Euclidean and SND estimates |
| `figure_S03` | Complete-run contributions for all raw metrics and layer/aggregation views | Main run-contribution table |
| `figure_S04` | Task accuracy; all metric contrasts; identity/parity/same-seed/different-seed endpoint groups | Task training, task estimates, task pair table and paired-block jackknife |
| `figure_S05` | Layerwise Pearson trajectories; separate within/between scatters; conditional final correlations | Main correlation estimates and raw W0 Alpha/CKA pairs |
| `figure_S06` | Raw Alpha changes/correlation under scaling; raw dataset contrast; logit/canonical-weight numerical errors | Gauge sensitivity/estimate/check tables |
| `figure_S07` | Full early accuracy trajectory and all five structural/representational contrasts | Early training and estimates |
| `figure_S08` | Pooled matching trajectories, assignment overlap, renumbering/function checks | Main pair tables; pooled run jackknife; alignment validation |
| `figure_S09` | Within/between activation distributions, improvement fractions, pooled trajectories | Main activation pairs and pooled run jackknife |
| `figure_S10` | Descriptive sign-retention trajectories and final sign-change run distributions | Main sign-retention table |

## Statistical and display conventions

- Pairwise uncertainty uses complete-run deletion, stratified by dataset. Paired task uncertainty deletes complete seed blocks. Pair counts never become the inferential sample size.
- Accuracy and early displacement bands use 95% Student-t intervals over independent runs; thin training lines show each run. Sign-retention trajectories and individual pair/run distributions remain descriptive.
- Main Figures 1–2 use direct canonical-Alpha profile matching. Figure 3 and S8–S9 use the candidate-selected benchmark, explicitly distinct from the primary matching procedure.
- Supplement S2, S3 and S5 use raw Alpha. Task comparisons use incoming weights only and no two-layer gauge.
- The early update axis uses `log2(s + 1)` and labels the actual update counts. Onset markers are computed from the first positive lower confidence limit.
- Gauge numerical-error plots place exact zeros at a documented display floor; the stored error values remain unchanged.
- Box/point displays show every observation. Horizontal marks indicate means; box summaries and point distributions are descriptive.
- Supplement section S11 contains Table S9, not an eleventh supplementary figure. Its synthetic multilayer result is produced by the validation pipeline and is not fabricated as an extra plot.
