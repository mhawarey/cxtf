#!/usr/bin/env python3
"""
CxTF Supplementary Experiment v2: Fixed Baselines + Statistical Significance
=============================================================================
Fixes from v1:
  - WLS-RW ISB: Reset ISB prior at trace boundaries to prevent cross-trace divergence;
    use actual time delta between epochs for random-walk variance; clamp ISB updates;
    convergence check per iteration; fall back to constant-ISB if solver diverges.
  - T-SPP: Removed learnable positional encoding (satellite order is arbitrary/meaningless).
    A constellation-blind transformer over an unordered set should have NO positional encoding.
    This is the correct/fair reimplementation.

Usage:
    python cxtf_supplementary_v2.py --data_dir ./data --output_dir ./results_12traces

Requires: torch, numpy, pandas, scipy
"""

import os
import sys
import math
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore', category=UserWarning)


# =====================================================================
# CONFIGURATION
# =====================================================================
class Config:
    d_obs = 10
    d_model = 128
    n_layers = 4
    n_heads = 8
    d_ff = 512
    n_constellations = 4
    max_sats = 50
    batch_size = 64
    lr = 1e-4
    weight_decay = 0.01
    warmup_steps = 1000
    max_epochs = 100
    patience = 10
    dropout = 0.1
    grad_clip = 1.0
    beta_loss = 0.5
    lambda_sparse = 0.001
    CONST_MAP = {1: 0, 6: 1, 5: 2, 3: 3}
    CONST_NAMES = {0: 'GPS', 1: 'Galileo', 2: 'BeiDou', 3: 'GLONASS'}
    PRIMARY_SIGNALS = ['GPS_L1_CA', 'GAL_E1_C_P', 'BDS_B1_I', 'GLO_G1_CA']
    min_sats_per_epoch = 8
    min_constellations = 2
    min_elevation = 5.0
    min_cn0 = 15.0


# =====================================================================
# DATA PREPROCESSING (identical to main experiment)
# =====================================================================
def lla_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f ** 2
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt_m) * np.sin(lat)
    return np.column_stack([x, y, z])


def load_and_preprocess(gnss_path, gt_path, config):
    gnss = pd.read_csv(gnss_path)
    gt = pd.read_csv(gt_path)
    gnss = gnss[gnss['SignalType'].isin(config.PRIMARY_SIGNALS)].copy()
    gnss = gnss[gnss['ConstellationType'].isin(config.CONST_MAP.keys())].copy()
    gnss['const_id'] = gnss['ConstellationType'].map(config.CONST_MAP)
    required_cols = [
        'utcTimeMillis', 'Svid', 'const_id',
        'RawPseudorangeMeters', 'AccumulatedDeltaRangeMeters',
        'Cn0DbHz', 'SvElevationDegrees', 'SvAzimuthDegrees',
        'PseudorangeRateMetersPerSecond',
        'SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters',
        'WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters',
    ]
    gnss = gnss.dropna(subset=required_cols)
    gnss['dx'] = gnss['SvPositionXEcefMeters'] - gnss['WlsPositionXEcefMeters']
    gnss['dy'] = gnss['SvPositionYEcefMeters'] - gnss['WlsPositionYEcefMeters']
    gnss['dz'] = gnss['SvPositionZEcefMeters'] - gnss['WlsPositionZEcefMeters']
    gnss['range_m'] = np.sqrt(gnss['dx'] ** 2 + gnss['dy'] ** 2 + gnss['dz'] ** 2)
    gnss['ux'] = gnss['dx'] / gnss['range_m']
    gnss['uy'] = gnss['dy'] / gnss['range_m']
    gnss['uz'] = gnss['dz'] / gnss['range_m']
    gnss['pr_residual'] = gnss['RawPseudorangeMeters'] - gnss['range_m']
    gnss = gnss[gnss['SvElevationDegrees'] >= config.min_elevation]
    gnss = gnss[gnss['Cn0DbHz'] >= config.min_cn0]
    epoch_stats = gnss.groupby('utcTimeMillis').agg(
        n_sats=('Svid', 'count'), n_const=('const_id', 'nunique'))
    valid_epochs = epoch_stats[
        (epoch_stats['n_sats'] >= config.min_sats_per_epoch) &
        (epoch_stats['n_const'] >= config.min_constellations)].index
    gnss = gnss[gnss['utcTimeMillis'].isin(valid_epochs)]
    gt_ecef = lla_to_ecef(
        gt['LatitudeDegrees'].values,
        gt['LongitudeDegrees'].values,
        gt['AltitudeMeters'].values)
    gt['gt_x'], gt['gt_y'], gt['gt_z'] = gt_ecef[:, 0], gt_ecef[:, 1], gt_ecef[:, 2]
    gt_times = gt[['UnixTimeMillis', 'gt_x', 'gt_y', 'gt_z']].rename(
        columns={'UnixTimeMillis': 'utcTimeMillis'})
    gnss = gnss.merge(gt_times, on='utcTimeMillis', how='inner')
    feature_cols = [
        'RawPseudorangeMeters', 'AccumulatedDeltaRangeMeters',
        'Cn0DbHz', 'SvElevationDegrees', 'SvAzimuthDegrees',
        'PseudorangeRateMetersPerSecond', 'ux', 'uy', 'uz', 'pr_residual',
    ]
    epochs_data = []
    for epoch_time, group in gnss.groupby('utcTimeMillis'):
        n_sats = len(group)
        if n_sats < config.min_sats_per_epoch:
            continue
        features = group[feature_cols].values.astype(np.float32)
        const_ids = group['const_id'].values.astype(np.int64)
        elevations = group['SvElevationDegrees'].values.astype(np.float32)
        azimuths = group['SvAzimuthDegrees'].values.astype(np.float32)
        wls_pos = group[['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters',
                         'WlsPositionZEcefMeters']].values[0].astype(np.float64)
        gt_pos = group[['gt_x', 'gt_y', 'gt_z']].values[0].astype(np.float64)
        delta_r = (gt_pos - wls_pos).astype(np.float32)
        epochs_data.append({
            'features': features, 'const_ids': const_ids,
            'elevations': elevations, 'azimuths': azimuths,
            'wls_pos': wls_pos, 'gt_pos': gt_pos,
            'delta_r': delta_r, 'n_sats': n_sats, 'epoch_time': epoch_time,
            'pseudoranges': group['RawPseudorangeMeters'].values.astype(np.float64),
            'sv_pos_x': group['SvPositionXEcefMeters'].values.astype(np.float64),
            'sv_pos_y': group['SvPositionYEcefMeters'].values.astype(np.float64),
            'sv_pos_z': group['SvPositionZEcefMeters'].values.astype(np.float64),
        })
    return epochs_data


# =====================================================================
# DATASET (identical to main experiment)
# =====================================================================
class GNSSDataset(Dataset):
    def __init__(self, epochs_data, config, stats=None):
        self.data = epochs_data
        self.config = config
        self.max_sats = config.max_sats
        if stats is None:
            all_f = np.concatenate([e['features'] for e in epochs_data], axis=0)
            self.mean = all_f.mean(axis=0)
            self.std = all_f.std(axis=0) + 1e-8
        else:
            self.mean, self.std = stats['mean'], stats['std']

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        n = min(e['n_sats'], self.max_sats)
        feat = (e['features'][:n] - self.mean) / self.std
        pf = np.zeros((self.max_sats, self.config.d_obs), dtype=np.float32)
        pf[:n] = feat
        pc = np.zeros(self.max_sats, dtype=np.int64)
        pc[:n] = e['const_ids'][:n]
        pe = np.zeros(self.max_sats, dtype=np.float32)
        pe[:n] = e['elevations'][:n]
        pa = np.zeros(self.max_sats, dtype=np.float32)
        pa[:n] = e['azimuths'][:n]
        mask = np.zeros(self.max_sats, dtype=np.float32)
        mask[:n] = 1.0
        return {
            'features': torch.tensor(pf),
            'const_ids': torch.tensor(pc),
            'elevations': torch.tensor(pe),
            'azimuths': torch.tensor(pa),
            'mask': torch.tensor(mask),
            'delta_r': torch.tensor(e['delta_r']),
            'n_sats': n,
        }


# =====================================================================
# CxTF MODEL (identical to main experiment — loaded from checkpoint)
# =====================================================================
class ElevPE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, el_deg):
        el_rad = el_deg.unsqueeze(-1) * (math.pi / 180.0)
        sincos = torch.cat([
            torch.sin(el_rad * self.inv_freq),
            torch.cos(el_rad * self.inv_freq),
        ], dim=-1)
        return sincos


class CxTF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.d_obs, config.d_model)
        self.const_emb = nn.Embedding(config.n_constellations, config.d_model)
        self.pe = ElevPE(config.d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.d_ff, dropout=config.dropout,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=config.n_layers)
        self.sel_head = nn.Linear(config.d_model, 1)
        self.pos_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2), nn.ReLU(),
            nn.Linear(config.d_model // 2, config.d_model // 4), nn.ReLU(),
            nn.Linear(config.d_model // 4, 3))
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.const_emb.weight, std=0.02)

    def forward(self, feat, cid, elev, mask, return_extras=False):
        z = self.proj(feat) + self.const_emb(cid) + self.pe(elev)
        z = self.encoder(z, src_key_padding_mask=(mask == 0))
        s = torch.sigmoid(self.sel_head(z).squeeze(-1)) * mask
        n_real = mask.sum(1, keepdim=True).clamp(min=1)
        sp = (s.sum(1) / n_real.squeeze()).mean()
        h = (s.unsqueeze(-1) * z).sum(1) / s.sum(1, keepdim=True).clamp(min=1e-8)
        dr = self.pos_mlp(h)
        out = {'delta_r': dr, 'scores': s, 'sparse_loss': sp}
        if return_extras:
            out['z'] = z
            out['emb'] = self.const_emb.weight.detach()
        return out


# =====================================================================
# T-SPP: FIXED — No positional encoding (satellite order is arbitrary)
# =====================================================================
class TSPP(nn.Module):
    """T-SPP reimplementation: constellation-blind transformer for SPP correction.

    Key design choices (fair comparison with CxTF):
      - NO constellation embeddings (constellation-blind)
      - NO positional encoding (satellite set is unordered; position-index PE
        would give the model spurious information from padding/batching order)
      - NO satellite selection (simple mean pooling)
      - Same d_model, n_layers, n_heads, d_ff, dropout as CxTF for fair comparison
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.d_obs, config.d_model)
        # NO positional encoding — satellite order is meaningless
        enc = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.d_ff, dropout=config.dropout,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=config.n_layers)
        self.pos_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2), nn.ReLU(),
            nn.Linear(config.d_model // 2, config.d_model // 4), nn.ReLU(),
            nn.Linear(config.d_model // 4, 3))
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, feat, cid, elev, mask, return_extras=False):
        # Project features only — no PE, no constellation embedding
        z = self.proj(feat)
        z = self.encoder(z, src_key_padding_mask=(mask == 0))
        # Mean pooling over valid tokens (no satellite selection)
        h = (mask.unsqueeze(-1) * z).sum(1) / mask.sum(1, keepdim=True).clamp(min=1e-8)
        dr = self.pos_mlp(h)
        return {'delta_r': dr, 'scores': mask, 'sparse_loss': torch.tensor(0.0)}


# =====================================================================
# WLS-SPP with Random-Walk ISB — FIXED
# =====================================================================
ISB_MAX = 1e6  # Clamp: any ISB beyond 1000 km is nonsense


def wls_solve_epoch(epoch_data, isb_prior=None, dt_seconds=1.0, q_spectral=0.01):
    """Weighted Least Squares SPP with random-walk ISB estimation.

    Fixes from v1:
      - Uses actual time delta (dt_seconds) for random-walk variance
      - Clamps ISB updates to prevent divergence
      - Convergence check: stops iterating if position update < 0.01 m
      - Falls back to no-prior solution if condition number is bad
    """
    pr = epoch_data['pseudoranges']
    sv_x = epoch_data['sv_pos_x']
    sv_y = epoch_data['sv_pos_y']
    sv_z = epoch_data['sv_pos_z']
    cids = epoch_data['const_ids']
    elevs = epoch_data['elevations']
    gt_pos = epoch_data['gt_pos']
    x0 = epoch_data['wls_pos'].copy()

    n_sats = len(pr)
    unique_const = np.unique(cids)
    non_gps = sorted([c for c in unique_const if c != 0])
    n_isb = len(non_gps)
    n_params = 4 + n_isb  # x, y, z, clk + ISBs

    # Elevation-dependent weights: w_i = sin^2(el_i)
    sin_el = np.sin(np.radians(np.maximum(elevs, 5.0)))
    w_diag = sin_el ** 2  # higher elevation = higher weight
    W = np.diag(w_diag)

    # Initialize ISB
    isb_est = {}
    for c in non_gps:
        if isb_prior is not None and c in isb_prior:
            isb_est[c] = np.clip(isb_prior[c], -ISB_MAX, ISB_MAX)
        else:
            isb_est[c] = 0.0

    pos = x0.copy()
    clk = 0.0

    # Random-walk variance: sigma^2 = q * dt
    sigma_rw_sq = q_spectral * max(dt_seconds, 0.1)

    # Iterative WLS — up to 10 iterations with convergence check
    for iteration in range(10):
        dx = sv_x - pos[0]
        dy = sv_y - pos[1]
        dz = sv_z - pos[2]
        ranges = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # Avoid division by zero
        ranges = np.maximum(ranges, 1.0)

        # Design matrix
        G = np.zeros((n_sats, n_params))
        G[:, 0] = -dx / ranges
        G[:, 1] = -dy / ranges
        G[:, 2] = -dz / ranges
        G[:, 3] = 1.0
        for idx, c in enumerate(non_gps):
            G[cids == c, 4 + idx] = 1.0

        # Predicted pseudoranges
        predicted = ranges + clk
        for idx, c in enumerate(non_gps):
            predicted[cids == c] += isb_est[c]

        dy_vec = pr - predicted

        try:
            GtWG = G.T @ W @ G
            GtWy = G.T @ W @ dy_vec

            # Add random-walk ISB prior constraint
            if isb_prior is not None:
                for idx, c in enumerate(non_gps):
                    if c in isb_prior:
                        constraint_weight = 1.0 / (sigma_rw_sq + 1e-10)
                        GtWG[4 + idx, 4 + idx] += constraint_weight
                        GtWy[4 + idx] += constraint_weight * (isb_prior[c] - isb_est[c])

            # Check condition number before solving
            cond = np.linalg.cond(GtWG)
            if cond > 1e12:
                # Badly conditioned — drop ISB prior constraints and solve fresh
                GtWG = G.T @ W @ G
                GtWy = G.T @ W @ dy_vec

            delta = np.linalg.solve(GtWG, GtWy)
        except np.linalg.LinAlgError:
            break

        # Clamp position updates to prevent runaway
        delta[:3] = np.clip(delta[:3], -1e5, 1e5)

        pos += delta[:3]
        clk += delta[3]
        for idx, c in enumerate(non_gps):
            isb_est[c] = np.clip(isb_est[c] + delta[4 + idx], -ISB_MAX, ISB_MAX)

        # Convergence check
        if np.sqrt(np.sum(delta[:3] ** 2)) < 0.01:
            break

    error_3d = np.sqrt(np.sum((pos - gt_pos) ** 2))
    error_2d = np.sqrt(np.sum((pos[:2] - gt_pos[:2]) ** 2))

    return pos, isb_est, error_3d, error_2d


def evaluate_wls_rw(test_data, trace_boundaries_test, q_spectral=0.01):
    """Evaluate WLS-SPP with random-walk ISB, resetting prior at trace boundaries.

    Args:
        test_data: list of epoch dicts (in original trace order)
        trace_boundaries_test: list of (start_idx, end_idx) tuples for each trace
            within test_data
        q_spectral: spectral density (m^2/s) for the random-walk ISB model
    """
    errors_3d = []
    errors_2d = []

    for tb_start, tb_end in trace_boundaries_test:
        trace_epochs = test_data[tb_start:tb_end]
        # Sort within trace by time
        trace_epochs = sorted(trace_epochs, key=lambda x: x['epoch_time'])

        isb_prior = None
        prev_time = None

        for epoch in trace_epochs:
            # Compute actual time delta
            if prev_time is not None:
                dt = (epoch['epoch_time'] - prev_time) / 1000.0  # ms -> seconds
                dt = max(dt, 0.1)  # minimum 0.1s
            else:
                dt = 1.0  # first epoch in trace

            _, isb_est, e3d, e2d = wls_solve_epoch(
                epoch, isb_prior=isb_prior, dt_seconds=dt, q_spectral=q_spectral)
            errors_3d.append(e3d)
            errors_2d.append(e2d)

            isb_prior = isb_est
            prev_time = epoch['epoch_time']

    errors_3d = np.array(errors_3d)
    errors_2d = np.array(errors_2d)

    return {
        'rmse_3d': float(np.sqrt((errors_3d ** 2).mean())),
        'rmse_2d': float(np.sqrt((errors_2d ** 2).mean())),
        'median_3d': float(np.median(errors_3d)),
        'p95_3d': float(np.percentile(errors_3d, 95)),
        'errors_3d': errors_3d,
        'errors_2d': errors_2d,
    }


# =====================================================================
# TRAINING & EVALUATION
# =====================================================================
def train_one_epoch(model, loader, opt, sched, config, device):
    model.train()
    total = 0
    for batch in loader:
        f = batch['features'].to(device)
        c = batch['const_ids'].to(device)
        e = batch['elevations'].to(device)
        m = batch['mask'].to(device)
        t = batch['delta_r'].to(device)
        opt.zero_grad()
        out = model(f, c, e, m)
        l1 = F.l1_loss(out['delta_r'], t)
        l2 = F.mse_loss(out['delta_r'], t)
        loss = ((1 - config.beta_loss) * l1 + config.beta_loss * l2
                + config.lambda_sparse * out.get('sparse_loss', torch.tensor(0.0)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt.step()
        sched.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        f = batch['features'].to(device)
        c = batch['const_ids'].to(device)
        e = batch['elevations'].to(device)
        m = batch['mask'].to(device)
        out = model(f, c, e, m)
        preds.append(out['delta_r'].cpu().numpy())
        trues.append(batch['delta_r'].numpy())
    p = np.concatenate(preds)
    t = np.concatenate(trues)
    e3d = np.sqrt(((p - t) ** 2).sum(1))
    e2d = np.sqrt(((p[:, :2] - t[:, :2]) ** 2).sum(1))
    return {
        'rmse_3d': float(np.sqrt((e3d ** 2).mean())),
        'rmse_2d': float(np.sqrt((e2d ** 2).mean())),
        'median_3d': float(np.median(e3d)),
        'p95_3d': float(np.percentile(e3d, 95)),
        'errors_3d': e3d,
        'errors_2d': e2d,
    }


def train_model(ModelClass, config, train_loader, val_loader, device, name="Model"):
    model = ModelClass(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.max_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)

    best_val, patience_ctr, best_state = float('inf'), 0, None
    for ep in range(config.max_epochs):
        loss = train_one_epoch(model, train_loader, opt, sched, config, device)
        val_m = evaluate_model(model, val_loader, device)
        if val_m['rmse_3d'] < best_val:
            best_val = val_m['rmse_3d']
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        if (ep + 1) % 10 == 0:
            print(f"    {name} ep{ep + 1:3d}: loss={loss:.4f} val_RMSE={val_m['rmse_3d']:.3f}m pat={patience_ctr}")
        if patience_ctr >= config.patience:
            print(f"    {name}: early stop ep{ep + 1}")
            break

    model.load_state_dict(best_state)
    model.to(device)
    return model, n_params


# =====================================================================
# STATISTICAL TESTS
# =====================================================================
def cohens_d(x, y):
    """Cohen's d for paired samples (x expected > y)."""
    diff = x - y
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def wilcoxon_test(errors_a, errors_b, name_a, name_b):
    """Wilcoxon signed-rank test: H1 = errors_a > errors_b (i.e., b is better)."""
    stat, p_raw = scipy_stats.wilcoxon(errors_a, errors_b, alternative='greater')
    d = cohens_d(errors_a, errors_b)
    if abs(d) < 0.2:
        d_label = "negligible"
    elif abs(d) < 0.5:
        d_label = "small"
    elif abs(d) < 0.8:
        d_label = "medium"
    else:
        d_label = "large"
    return {
        'comparison': f"{name_a} vs. {name_b}",
        'W': int(stat),
        'p_raw': float(p_raw),
        'd': float(d),
        'd_label': d_label,
    }


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="CxTF Supplementary v2")
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--output_dir', default='./results_12traces')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    config = Config()
    t0 = time.time()

    print("=" * 70)
    print("CxTF SUPPLEMENTARY v2: Fixed Baselines + Statistical Tests")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    print("=" * 70)

    # ================================================================
    # [1/6] LOAD DATA
    # ================================================================
    print("\n[1/6] Loading data...")
    all_epochs = []
    trace_names = []
    for trace_dir in sorted(Path(args.data_dir).iterdir()):
        if trace_dir.is_dir():
            gp = trace_dir / 'device_gnss.csv'
            tp = trace_dir / 'ground_truth.csv'
            if gp.exists() and tp.exists():
                print(f"  Loading {trace_dir.name}...")
                data = load_and_preprocess(str(gp), str(tp), config)
                # Tag each epoch with its trace name for boundary detection
                for ep in data:
                    ep['trace_name'] = trace_dir.name
                all_epochs.extend(data)
                trace_names.append(trace_dir.name)

    all_c = set()
    for e in all_epochs:
        all_c.update(e['const_ids'].tolist())
    config.n_constellations = max(all_c) + 1
    print(f"  Total: {len(all_epochs)} epochs, {len(all_c)} constellations, "
          f"{len(trace_names)} traces")

    # ================================================================
    # SPLIT — per-trace 70/15/15 (identical to main experiment)
    # ================================================================
    import random
    random.seed(42)

    train_data, val_data, test_data = [], [], []
    test_trace_boundaries = []  # (start_idx, end_idx) within test_data

    # Detect trace boundaries
    trace_boundaries = []
    trace_start = 0
    for i in range(1, len(all_epochs)):
        if all_epochs[i]['trace_name'] != all_epochs[i - 1]['trace_name']:
            trace_boundaries.append((trace_start, i))
            trace_start = i
    trace_boundaries.append((trace_start, len(all_epochs)))
    print(f"  Detected {len(trace_boundaries)} traces for per-trace splitting")

    for start, end in trace_boundaries:
        trace_epochs = all_epochs[start:end]
        nt = len(trace_epochs)
        n_tr = int(0.70 * nt)
        n_va = int(0.15 * nt)
        test_start_idx = len(test_data)
        train_data.extend(trace_epochs[:n_tr])
        val_data.extend(trace_epochs[n_tr:n_tr + n_va])
        test_data.extend(trace_epochs[n_tr + n_va:])
        test_end_idx = len(test_data)
        if test_end_idx > test_start_idx:
            test_trace_boundaries.append((test_start_idx, test_end_idx))

    random.shuffle(train_data)
    print(f"  Split: train={len(train_data)} val={len(val_data)} test={len(test_data)}")
    print(f"  Test trace segments: {len(test_trace_boundaries)}")

    train_ds = GNSSDataset(train_data, config)
    stats = train_ds.get_stats()
    val_ds = GNSSDataset(val_data, config, stats=stats)
    test_ds = GNSSDataset(test_data, config, stats=stats)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, num_workers=2)

    # ================================================================
    # [2/6] LOAD TRAINED CxTF
    # ================================================================
    print("\n[2/6] Loading trained CxTF model...")
    cxtf_model = CxTF(config).to(device)
    model_path = os.path.join(args.output_dir, 'cxtf_best.pt')
    cxtf_model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))
    cxtf_m = evaluate_model(cxtf_model, test_loader, device)
    print(f"  CxTF: 3D RMSE = {cxtf_m['rmse_3d']:.3f} m")

    # ================================================================
    # [3/6] WLS-SPP CONSTANT ISB (per-epoch errors for stats)
    # ================================================================
    print("\n[3/6] WLS-SPP (constant ISB) per-epoch errors...")
    wls_errors_3d = np.array([np.sqrt((e['delta_r'] ** 2).sum()) for e in test_data])
    wls_errors_2d = np.array([np.sqrt((e['delta_r'][:2] ** 2).sum()) for e in test_data])
    wls_m = {
        'rmse_3d': float(np.sqrt((wls_errors_3d ** 2).mean())),
        'rmse_2d': float(np.sqrt((wls_errors_2d ** 2).mean())),
        'median_3d': float(np.median(wls_errors_3d)),
        'p95_3d': float(np.percentile(wls_errors_3d, 95)),
        'errors_3d': wls_errors_3d,
        'errors_2d': wls_errors_2d,
    }
    print(f"  WLS-SPP (const ISB): 3D RMSE = {wls_m['rmse_3d']:.3f} m")

    # ================================================================
    # [4/6] BASELINE 2: WLS-SPP RANDOM-WALK ISB (FIXED)
    # ================================================================
    print("\n[4/6] Baseline 2: WLS-SPP (random-walk ISB) — trace-aware...")
    rw_m = evaluate_wls_rw(test_data, test_trace_boundaries, q_spectral=0.01)
    rw_imp = (1 - rw_m['rmse_3d'] / wls_m['rmse_3d']) * 100
    print(f"  WLS-SPP (RW ISB): 3D RMSE = {rw_m['rmse_3d']:.3f} m "
          f"({rw_imp:+.1f}% vs const ISB)")

    # Sanity check: if RW ISB result is wildly wrong, flag it
    if rw_m['rmse_3d'] > 100:
        print(f"  *** WARNING: RW ISB RMSE={rw_m['rmse_3d']:.1f}m — still divergent! ***")
    elif abs(rw_imp) < 0.1:
        print(f"  Note: RW ISB ≈ constant ISB on this dataset (expected for short traces)")

    # ================================================================
    # [5/6] BASELINE 3: T-SPP (FIXED — NO POSITIONAL ENCODING)
    # ================================================================
    print("\n[5/6] Baseline 3: T-SPP (constellation-blind, NO PE)...")
    tspp_model, tspp_params = train_model(
        TSPP, config, train_loader, val_loader, device, "T-SPP")
    tspp_m = evaluate_model(tspp_model, test_loader, device)
    tspp_imp = (1 - tspp_m['rmse_3d'] / wls_m['rmse_3d']) * 100
    cxtf_vs_tspp = (1 - cxtf_m['rmse_3d'] / tspp_m['rmse_3d']) * 100
    print(f"  T-SPP: 3D RMSE = {tspp_m['rmse_3d']:.3f} m "
          f"({tspp_imp:.1f}% vs WLS)")
    print(f"  CxTF vs T-SPP: {cxtf_vs_tspp:+.1f}%")

    # ================================================================
    # [6/6] STATISTICAL SIGNIFICANCE TESTS
    # ================================================================
    print("\n[6/6] Statistical significance tests...")

    comparisons = []
    n_comparisons = 3  # WLS-const, WLS-RW, T-SPP (3 pairwise tests)

    # 1. CxTF vs WLS-SPP (constant ISB)
    n_min = min(len(cxtf_m['errors_3d']), len(wls_errors_3d))
    comparisons.append(wilcoxon_test(
        wls_errors_3d[:n_min], cxtf_m['errors_3d'][:n_min],
        "WLS-SPP (const ISB)", "CxTF"))

    # 2. CxTF vs WLS-SPP (RW ISB) — only if RW didn't diverge
    if rw_m['rmse_3d'] < 100:
        n_min2 = min(len(cxtf_m['errors_3d']), len(rw_m['errors_3d']))
        comparisons.append(wilcoxon_test(
            rw_m['errors_3d'][:n_min2], cxtf_m['errors_3d'][:n_min2],
            "WLS-SPP (RW ISB)", "CxTF"))
    else:
        print("  Skipping WLS-RW stat test (divergent)")

    # 3. CxTF vs T-SPP
    n_min3 = min(len(cxtf_m['errors_3d']), len(tspp_m['errors_3d']))
    comparisons.append(wilcoxon_test(
        tspp_m['errors_3d'][:n_min3], cxtf_m['errors_3d'][:n_min3],
        "T-SPP", "CxTF"))

    # Bonferroni correction
    for c in comparisons:
        c['p_corrected'] = min(c['p_raw'] * n_comparisons, 1.0)
        c['significant'] = c['p_corrected'] < 0.05

    # ================================================================
    # PRINT RESULTS
    # ================================================================
    print("\n" + "=" * 80)
    print("SUPPLEMENTARY v2 RESULTS")
    print("=" * 80)

    print(f"\n{'Method':<30} {'2D RMSE':>9} {'3D RMSE':>9} "
          f"{'Median':>9} {'95th':>9} {'Improv':>8}")
    print("-" * 80)

    def fmt_row(name, m, baseline_rmse=None):
        imp = "—" if baseline_rmse is None else \
              f"{(1 - m['rmse_3d'] / baseline_rmse) * 100:.1f}%"
        print(f"{name:<30} {m['rmse_2d']:>8.2f}m {m['rmse_3d']:>8.2f}m "
              f"{m['median_3d']:>8.2f}m {m['p95_3d']:>8.2f}m {imp:>8}")

    fmt_row("WLS-SPP (const ISB)", wls_m)
    if rw_m['rmse_3d'] < 100:
        fmt_row("WLS-SPP (random-walk ISB)", rw_m, wls_m['rmse_3d'])
    else:
        print(f"{'WLS-SPP (random-walk ISB)':<30} {'*** DIVERGENT — excluded ***'}")
    fmt_row("T-SPP (Wu et al., 2024)", tspp_m, wls_m['rmse_3d'])
    fmt_row("CxTF (proposed)", cxtf_m, wls_m['rmse_3d'])

    print(f"\n{'Comparison':<35} {'W':>12} {'p(raw)':>12} "
          f"{'p(corr)':>12} {'d':>12} {'Sig?':>6}")
    print("-" * 95)
    for c in comparisons:
        ps = f"{c['p_raw']:.2e}" if c['p_raw'] < 0.001 else f"{c['p_raw']:.4f}"
        pc = f"{c['p_corrected']:.2e}" if c['p_corrected'] < 0.001 else \
             f"{c['p_corrected']:.4f}"
        sig = "Yes" if c['significant'] else "No"
        print(f"{c['comparison']:<35} {c['W']:>12,} {ps:>12} {pc:>12} "
              f"{c['d']:>7.2f} ({c['d_label']:<10}) {sig:>4}")

    print(f"\nCxTF vs T-SPP: {cxtf_vs_tspp:+.1f}%")
    if rw_m['rmse_3d'] < 100:
        print(f"CxTF vs WLS-RW: "
              f"{(1 - cxtf_m['rmse_3d'] / rw_m['rmse_3d']) * 100:+.1f}%")

    # ================================================================
    # SAVE
    # ================================================================
    results = {
        'wls_const_isb': {k: v for k, v in wls_m.items()
                          if k not in ('errors_3d', 'errors_2d')},
        'wls_rw_isb': {k: v for k, v in rw_m.items()
                       if k not in ('errors_3d', 'errors_2d')},
        'wls_rw_isb_divergent': rw_m['rmse_3d'] > 100,
        'tspp': {k: v for k, v in tspp_m.items()
                 if k not in ('errors_3d', 'errors_2d')},
        'tspp_params': tspp_params,
        'cxtf': {k: v for k, v in cxtf_m.items()
                 if k not in ('errors_3d', 'errors_2d')},
        'statistical_tests': comparisons,
        'bonferroni_n': n_comparisons,
        'elapsed_seconds': time.time() - t0,
        'script_version': 'v2',
        'fixes': [
            'WLS-RW: reset ISB prior at trace boundaries, actual dt, clamping, convergence check',
            'T-SPP: removed learnable positional encoding (satellite order is arbitrary)',
        ],
    }

    out_path = os.path.join(args.output_dir, 'supplementary_results_v2.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    txt_path = os.path.join(args.output_dir, 'supplementary_summary_v2.txt')
    with open(txt_path, 'w') as f:
        f.write("SUPPLEMENTARY v2 RESULTS\n" + "=" * 80 + "\n\n")
        f.write("Fixes applied:\n")
        for fix in results['fixes']:
            f.write(f"  - {fix}\n")
        f.write(f"\n--- Table 4 additions ---\n")
        if rw_m['rmse_3d'] < 100:
            f.write(f"WLS-SPP (random-walk ISB): 2D={rw_m['rmse_2d']:.2f} "
                    f"3D={rw_m['rmse_3d']:.2f} Med={rw_m['median_3d']:.2f} "
                    f"95th={rw_m['p95_3d']:.2f} "
                    f"Imp={(1 - rw_m['rmse_3d'] / wls_m['rmse_3d']) * 100:.1f}%\n")
        else:
            f.write("WLS-SPP (random-walk ISB): DIVERGENT — excluded\n")
        f.write(f"T-SPP (Wu et al., 2024):   2D={tspp_m['rmse_2d']:.2f} "
                f"3D={tspp_m['rmse_3d']:.2f} Med={tspp_m['median_3d']:.2f} "
                f"95th={tspp_m['p95_3d']:.2f} "
                f"Imp={(1 - tspp_m['rmse_3d'] / wls_m['rmse_3d']) * 100:.1f}%\n\n")
        f.write("--- Table 5 ---\n")
        for c in comparisons:
            f.write(f"{c['comparison']}: W={c['W']:,}, "
                    f"p_raw={c['p_raw']:.2e}, p_corr={c['p_corrected']:.2e}, "
                    f"d={c['d']:.2f} ({c['d_label']}), sig={c['significant']}\n")

    print(f"\nSaved: {out_path}")
    print(f"Saved: {txt_path}")
    print(f"Total time: {time.time() - t0:.0f}s")
    print("\n>>> Run complete. Upload supplementary_results_v2.json to update the paper. <<<")


if __name__ == '__main__':
    main()
