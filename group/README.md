# Group Analysis

Group-level analysis of physiological data from 10 participants across 3 sessions. Focuses on HRV, pupil dilation, and psychometric test timing — pulling out summary statistics and looking at correlations between modalities.

## What's measured

| Metric | Source | What it captures |
|--------|--------|------------------|
| HRV SDNN | Polar H10+ (inter-beat intervals) | Cardiac autonomic regulation — standard deviation of NN intervals |
| Pupil Dilation STD | Pupil Labs Core | Variability in pupil diameter, indexes cognitive load and arousal |
| Test Duration STD | Psychometric app timestamps | Response time variability across test items |

## Notebooks

```
├── group_analysis.ipynb              # Full-cohort psychometric + physiology analysis
├── group_correlation_matrix.ipynb    # Cross-modal correlation matrix (FDR-corrected)
├── group_correlation_heatmap.ipynb   # Generates the README headline heatmap PNG
├── group_hrv.ipynb                   # Heart-rate variability across sessions
├── group_eye.ipynb                   # Pupil dilation across sessions
├── group_duration.ipynb              # Test-completion-time variability
├── group_sd.ipynb                    # Standard-deviation metrics
├── group_time.ipynb                  # Response-timing analysis
└── duration.ipynb                    # Per-session duration breakdowns
```

Summary CSVs live in `../data/group_results/`. See
[`../DATA_PROVENANCE.md`](../DATA_PROVENANCE.md) for how they are generated and
which rows are regenerable from committed data.

## Analysis approach

1. **Aggregation** — Each notebook reads per-participant summary CSVs from `data/group_results/`
2. **Group statistics** — Compute means, standard deviations, and ranges across all participants and sessions
3. **Correlation** — Cross-modal analysis looking at relationships between HRV, pupil dilation, and response timing
