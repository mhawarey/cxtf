# CxTF: Constellation-Aware Transformer for Multi-GNSS Positioning

Reference implementation for the paper:

> **Hawarey, M. (2026).** A Constellation-Aware Transformer Architecture for Multi-GNSS Positioning: Learned Inter-System Bias Estimation and Attention-Based Satellite Selection. *AIR Journal of Engineering & Technology*, Vol. 2026.

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

Default configuration: d_model=128, L=4, H=8, ~1.6M parameters.

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

This implementation uses the [Google Smartphone Decimeter Challenge (GSDC) 2023–2024](https://www.kaggle.com/competitions/smartphone-decimeter-2023/data) dataset. Download from Kaggle and organize as:

```
data/
  2020-06-25-00-34/
    device_gnss.csv
    ground_truth.csv
  2021-01-04-21-50/
    device_gnss.csv
    ground_truth.csv
```

Each trace folder should contain `device_gnss.csv` (raw GNSS measurements) and `ground_truth.csv` (NovAtel SPAN reference positions).

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

## Preliminary Results (2 traces, CPU training)

| Method | 3D RMSE (m) | Median (m) | 95th %ile (m) | Improvement |
|--------|-------------|------------|----------------|-------------|
| WLS-SPP (baseline) | 5.15 | 3.65 | 8.63 | — |
| CxTF (proposed) | 4.79 | 2.92 | 8.66 | 6.9% (RMSE) / 20.0% (median) |
| Ablation A (no emb.) | 4.73 | 2.90 | 8.40 | 8.2% |
| Ablation B (no sel.) | 4.77 | 2.91 | 8.80 | 7.4% |
| Ablation C (no PE) | 4.96 | 3.15 | 9.23 | 3.7% |

> **Cross-environment note:** An exploratory 5-trace experiment (8,517 epochs, 2020–2023) showed that CxTF performance degrades under cross-environment domain shift when trained with limited data on CPU. The paper (Section 5.4, Limitation 5) discusses this finding in detail and identifies GPU-scale training on the full GSDC corpus (150+ traces) with per-trace cross-validation as the immediate research priority.

## Citation

```bibtex
@article{hawarey2026cxtf,
  title={A Constellation-Aware Transformer Architecture for Multi-GNSS Positioning: Learned Inter-System Bias Estimation and Attention-Based Satellite Selection},
  author={Hawarey, Mosab},
  journal={AIR Journal of Engineering \& Technology},
  volume={2026},
  year={2026},
  doi={10.65737/AIRJET2026XXX}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Mosab Hawarey** — [Geospatial Research](https://geospatial.ch) (https://hawarey.org/mosab)
