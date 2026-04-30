# CxTF: Constellation-Aware Transformer for Multi-GNSS Positioning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reference implementation for the paper:

**Hawarey, M. (2026).** A Constellation-Aware Transformer Architecture for Multi-GNSS Positioning: Learned Inter-System Bias Estimation and Attention-Based Satellite Selection. *AIR Journal of Engineering and Technology*, Vol. 2026, AIRJET2026613.
https://doi.org/10.65737/AIRJET2026613


>
> Mosab Hawarey
>
> *AIR Journal of Engineering and Technology*, Vol. 2026
>
> Journal DOI: 10.65737/AIRJET
>
> Publisher: Artificial Intelligence Review AIR Publishing House LLC
>
> Article ID: AIRJET2026613
> 
> Article DOI: https://doi.org/10.65737/AIRJET2026613

## Original Paper

https://doi.org/10.65737/AIRJET2026613
>
https://airjournals.org/doi/10.65737.AIRJET2026613.html

## Overview

CxTF is a transformer-based architecture for multi-constellation GNSS single-point positioning that introduces three innovations:

1. **Constellation embeddings** — learnable vector representations for GPS, Galileo, BeiDou, and GLONASS
2. **Cross-constellation attention** — implicit inter-system bias (ISB) learning via the natural block structure of the attention matrix
3. **Attention-based satellite selection** — learned importance scores jointly optimizing geometry, signal quality, and constellation balance

## Architecture

- Input: 10-dimensional per-satellite feature vectors (pseudorange, carrier phase, C/N₀, elevation, azimuth, Doppler, direction vector, residual)
- Constellation embedding + elevation positional encoding
- L-layer transformer encoder with multi-head self-attention
- Sigmoid-gated satellite selection with sparsity regularization
- Position correction MLP outputting Δr ∈ ℝ³

Default configuration: d_model=128, L=4, H=8, ~800K parameters.

## Requirements

```
Python >= 3.10
torch >= 2.0
pandas
numpy
scikit-learn
matplotlib
```

Install:
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## Data

This implementation uses the [Google Smartphone Decimeter Challenge (GSDC) 2023–2024](https://www.kaggle.com/competitions/smartphone-decimeter-2023/data) dataset. Download Pixel 7 Pro traces from Kaggle and organize as:

```
data/
  2023-05-09-21-32/
    device_gnss.csv
    ground_truth.csv
  2023-05-16-19-55/
    device_gnss.csv
    ground_truth.csv
  ...
```

The published results use 12 Pixel 7 Pro traces (7 from May 2023, 5 from September 2023). Each trace folder needs only `device_gnss.csv` (raw GNSS measurements) and `ground_truth.csv` (NovAtel SPAN reference positions). The script auto-discovers all trace folders under `data/`.

## Usage

```bash
python cxtf_full_experiment.py --data_dir ./data --output_dir ./results
```

Options:
- `--data_dir` — path to data directory (default: `./data`)
- `--output_dir` — path for results (default: `./results`)
- `--device` — `auto`, `cpu`, or `cuda` (default: `auto`)

The script will:
1. Load and preprocess all traces (6-step pipeline per paper Section 4.1)
2. Evaluate the WLS-SPP baseline
3. Train the full CxTF model with early stopping
4. Train three ablation variants (no embeddings, no selection, no PE)
5. Run interpretability analysis (embedding geometry, selection patterns)
6. Generate publication figures and save results

## Supplementary Experiment

`cxtf_supplementary.py` runs the additional baselines (T-SPP, WLS random-walk ISB) and statistical significance tests (Wilcoxon signed-rank, Cohen's d) reported in Tables 5–6 of the paper. It loads the trained CxTF checkpoint from the main experiment:
```bash
python cxtf_supplementary.py --data_dir ./data --output_dir ./results
```

## Output

```
results/
  results.json          — All metrics in JSON format
  results_summary.txt   — Formatted results table
  cxtf_best.pt          — Best model weights
  fig_cdf.png           — Position error CDF curves
  fig_embeddings.png    — Learned constellation embedding analysis
  fig_selection.png     — Satellite selection score analysis
```

## Empirical Results (12 Pixel 7 Pro traces, 18,676 epochs, CPU training)

| Method | 3D RMSE (m) | Median (m) | 95th %ile (m) | Improvement |
|--------|-------------|------------|----------------|-------------|
| WLS-SPP (baseline) | 6.68 | 5.15 | 11.15 | — |
| CxTF (proposed) | 5.30 | 3.59 | 8.73 | 20.6% (RMSE) / 30.3% (median) |
| Ablation A (no emb.) | 5.49 | 3.64 | 9.24 | 17.8% |
| Ablation B (no sel.) | 5.25 | 3.48 | 8.50 | 21.5% |
| Ablation C (no PE) | 5.44 | 3.63 | 8.83 | 18.6% |

Validation uses 12 GSDC traces from a single receiver type (Google Pixel 7 Pro), split per-trace (70/15/15%) to prevent data leakage. See the paper (Section 4.4.1) for full details.

## Author

**Dr. Mosab Hawarey**
>
PhD, Geodetic & Photogrammetric Engineering (ITU) | MSc, Geomatics (Purdue) | MBA (Wales) | BSc, MSc (METU)

- GitHub: https://github.com/mhawarey
- Personal: https://hawarey.org/mosab
- ORCID: https://orcid.org/0000-0001-7846-951X

## License

MIT License.



