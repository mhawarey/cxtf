#!/usr/bin/env python3
"""
CxTF: Constellation-Aware Transformer for Multi-GNSS Positioning
Full Experiment Script - Run locally with GPU

Usage:
    pip install torch pandas numpy scikit-learn matplotlib
    python cxtf_full_experiment.py --data_dir ./data --output_dir ./results

Expected data structure:
    data/
      2020-06-25-00-34/
        device_gnss.csv
        ground_truth.csv
      2021-01-04-21-50/
        device_gnss.csv
        ground_truth.csv

Outputs:
    results/
      results.json          - All metrics
      results_summary.txt   - Formatted table
      cxtf_best.pt          - Best model weights
      fig_cdf.png           - Position error CDF
      fig_attention.png     - Attention block analysis
      fig_embeddings.png    - Constellation embedding PCA
      fig_selection.png     - Satellite selection skyplot
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

warnings.filterwarnings('ignore', category=UserWarning)


# =====================================================================
# CONFIGURATION (Paper Tables 1-3)
# =====================================================================
class Config:
    # Architecture (Table 1) - FULL SIZE
    d_obs = 10
    d_model = 128
    n_layers = 4
    n_heads = 8
    d_ff = 512
    n_constellations = 4  # Auto-detected
    max_sats = 50

    # Training (Table 3)
    batch_size = 64
    lr = 1e-4
    weight_decay = 0.01
    warmup_steps = 1000
    max_epochs = 100
    patience = 10
    dropout = 0.1
    grad_clip = 1.0
    beta_loss = 0.5
    lambda_sparse = 0.01

    CONST_MAP = {1: 0, 6: 1, 5: 2, 3: 3}  # GPS, Galileo, BeiDou, GLONASS
    CONST_NAMES = {0: 'GPS', 1: 'Galileo', 2: 'BeiDou', 3: 'GLONASS'}
    PRIMARY_SIGNALS = ['GPS_L1_CA', 'GAL_E1_C_P', 'BDS_B1_I', 'GLO_G1_CA']
    min_sats_per_epoch = 8
    min_constellations = 2
    min_elevation = 5.0
    min_cn0 = 15.0


# =====================================================================
# DATA PREPROCESSING (Section 4.1)
# =====================================================================
def lla_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2*f - f**2
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt_m) * np.sin(lat)
    return np.column_stack([x, y, z])


def load_and_preprocess(gnss_path, gt_path, config):
    print(f"  Loading {Path(gnss_path).parent.name}...")
    gnss = pd.read_csv(gnss_path)
    gt = pd.read_csv(gt_path)

    # Filter primary signals and known constellations
    gnss = gnss[gnss['SignalType'].isin(config.PRIMARY_SIGNALS)].copy()
    gnss = gnss[gnss['ConstellationType'].isin(config.CONST_MAP.keys())].copy()
    gnss['const_id'] = gnss['ConstellationType'].map(config.CONST_MAP)

    required = ['utcTimeMillis', 'Svid', 'const_id',
                'RawPseudorangeMeters', 'AccumulatedDeltaRangeMeters',
                'Cn0DbHz', 'SvElevationDegrees', 'SvAzimuthDegrees',
                'PseudorangeRateMetersPerSecond',
                'SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters',
                'WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']
    gnss = gnss.dropna(subset=[c for c in required if c in gnss.columns])

    # Direction vectors and residuals
    gnss['dx'] = gnss['SvPositionXEcefMeters'] - gnss['WlsPositionXEcefMeters']
    gnss['dy'] = gnss['SvPositionYEcefMeters'] - gnss['WlsPositionYEcefMeters']
    gnss['dz'] = gnss['SvPositionZEcefMeters'] - gnss['WlsPositionZEcefMeters']
    gnss['range_m'] = np.sqrt(gnss['dx']**2 + gnss['dy']**2 + gnss['dz']**2)
    gnss['ux'] = gnss['dx'] / gnss['range_m']
    gnss['uy'] = gnss['dy'] / gnss['range_m']
    gnss['uz'] = gnss['dz'] / gnss['range_m']
    gnss['pr_residual'] = gnss['RawPseudorangeMeters'] - gnss['range_m']

    # Quality filters
    gnss = gnss[gnss['SvElevationDegrees'] >= config.min_elevation]
    gnss = gnss[gnss['Cn0DbHz'] >= config.min_cn0]

    epoch_stats = gnss.groupby('utcTimeMillis').agg(
        n_sats=('Svid', 'count'), n_const=('const_id', 'nunique'))
    valid_epochs = epoch_stats[
        (epoch_stats['n_sats'] >= config.min_sats_per_epoch) &
        (epoch_stats['n_const'] >= config.min_constellations)].index
    gnss = gnss[gnss['utcTimeMillis'].isin(valid_epochs)]

    # Ground truth
    gt_ecef = lla_to_ecef(gt['LatitudeDegrees'].values,
                          gt['LongitudeDegrees'].values,
                          gt['AltitudeMeters'].values)
    gt['gt_x'], gt['gt_y'], gt['gt_z'] = gt_ecef[:,0], gt_ecef[:,1], gt_ecef[:,2]
    gt_times = gt[['UnixTimeMillis','gt_x','gt_y','gt_z']].rename(
        columns={'UnixTimeMillis': 'utcTimeMillis'})
    gnss = gnss.merge(gt_times, on='utcTimeMillis', how='inner')

    feature_cols = ['RawPseudorangeMeters', 'AccumulatedDeltaRangeMeters',
                    'Cn0DbHz', 'SvElevationDegrees', 'SvAzimuthDegrees',
                    'PseudorangeRateMetersPerSecond', 'ux', 'uy', 'uz', 'pr_residual']

    epochs_data = []
    for epoch_time, group in gnss.groupby('utcTimeMillis'):
        n_sats = len(group)
        if n_sats < config.min_sats_per_epoch:
            continue
        features = group[feature_cols].values.astype(np.float32)
        const_ids = group['const_id'].values.astype(np.int64)
        elevations = group['SvElevationDegrees'].values.astype(np.float32)
        azimuths = group['SvAzimuthDegrees'].values.astype(np.float32)
        wls_pos = group[['WlsPositionXEcefMeters','WlsPositionYEcefMeters',
                         'WlsPositionZEcefMeters']].values[0].astype(np.float64)
        gt_pos = group[['gt_x','gt_y','gt_z']].values[0].astype(np.float64)
        delta_r = (gt_pos - wls_pos).astype(np.float32)

        epochs_data.append({
            'features': features, 'const_ids': const_ids,
            'elevations': elevations, 'azimuths': azimuths,
            'wls_pos': wls_pos, 'gt_pos': gt_pos,
            'delta_r': delta_r, 'n_sats': n_sats, 'epoch_time': epoch_time,
        })

    const_present = set()
    for e in epochs_data:
        const_present.update(e['const_ids'].tolist())
    print(f"    {len(epochs_data)} epochs, {len(const_present)} constellations "
          f"({[config.CONST_NAMES.get(c,'?') for c in sorted(const_present)]})")
    return epochs_data


# =====================================================================
# DATASET
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
        pa[:n] = e['azimuths'][:n] if 'azimuths' in e else 0
        mask = np.zeros(self.max_sats, dtype=np.float32)
        mask[:n] = 1.0
        return {
            'features': torch.tensor(pf), 'const_ids': torch.tensor(pc),
            'elevations': torch.tensor(pe), 'azimuths': torch.tensor(pa),
            'mask': torch.tensor(mask), 'delta_r': torch.tensor(e['delta_r']),
            'n_sats': n,
        }


# =====================================================================
# MODEL
# =====================================================================
class ElevPE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, el_deg):
        el_rad = el_deg.unsqueeze(-1) * (math.pi / 180.0)  # (B, N, 1)
        sincos = torch.cat([
            torch.sin(el_rad * self.inv_freq),
            torch.cos(el_rad * self.inv_freq)
        ], dim=-1)  # (B, N, d)
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


# Ablation variants
class CxTF_NoEmb(CxTF):
    def forward(self, feat, cid, elev, mask, **kw):
        z = self.proj(feat) + self.pe(elev)
        z = self.encoder(z, src_key_padding_mask=(mask == 0))
        s = torch.sigmoid(self.sel_head(z).squeeze(-1)) * mask
        sp = (s.sum(1) / mask.sum(1, keepdim=True).clamp(min=1).squeeze()).mean()
        h = (s.unsqueeze(-1) * z).sum(1) / s.sum(1, keepdim=True).clamp(min=1e-8)
        return {'delta_r': self.pos_mlp(h), 'scores': s, 'sparse_loss': sp}

class CxTF_NoSel(CxTF):
    def forward(self, feat, cid, elev, mask, **kw):
        z = self.proj(feat) + self.const_emb(cid) + self.pe(elev)
        z = self.encoder(z, src_key_padding_mask=(mask == 0))
        h = (mask.unsqueeze(-1) * z).sum(1) / mask.sum(1, keepdim=True).clamp(min=1e-8)
        return {'delta_r': self.pos_mlp(h), 'scores': mask, 'sparse_loss': torch.tensor(0.0)}

class CxTF_NoPE(CxTF):
    def forward(self, feat, cid, elev, mask, **kw):
        z = self.proj(feat) + self.const_emb(cid)
        z = self.encoder(z, src_key_padding_mask=(mask == 0))
        s = torch.sigmoid(self.sel_head(z).squeeze(-1)) * mask
        sp = (s.sum(1) / mask.sum(1, keepdim=True).clamp(min=1).squeeze()).mean()
        h = (s.unsqueeze(-1) * z).sum(1) / s.sum(1, keepdim=True).clamp(min=1e-8)
        return {'delta_r': self.pos_mlp(h), 'scores': s, 'sparse_loss': sp}


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
        loss = (1-config.beta_loss)*l1 + config.beta_loss*l2 + config.lambda_sparse*out['sparse_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt.step()
        sched.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues, scores_all, cids_all, masks_all, elevs_all, azs_all = [], [], [], [], [], [], []
    for batch in loader:
        f = batch['features'].to(device)
        c = batch['const_ids'].to(device)
        e = batch['elevations'].to(device)
        m = batch['mask'].to(device)
        out = model(f, c, e, m, return_extras=True)
        preds.append(out['delta_r'].cpu().numpy())
        trues.append(batch['delta_r'].numpy())
        scores_all.append(out['scores'].cpu().numpy())
        cids_all.append(c.cpu().numpy())
        masks_all.append(m.cpu().numpy())
        elevs_all.append(e.cpu().numpy())
        azs_all.append(batch['azimuths'].numpy())

    p = np.concatenate(preds); t = np.concatenate(trues)
    e3d = np.sqrt(((p - t)**2).sum(1))
    e2d = np.sqrt(((p[:,:2] - t[:,:2])**2).sum(1))
    metrics = {
        'rmse_3d': float(np.sqrt((e3d**2).mean())),
        'rmse_2d': float(np.sqrt((e2d**2).mean())),
        'median_3d': float(np.median(e3d)),
        'p95_3d': float(np.percentile(e3d, 95)),
    }
    details = {
        'errors_3d': e3d, 'errors_2d': e2d,
        'scores': np.concatenate(scores_all),
        'const_ids': np.concatenate(cids_all),
        'masks': np.concatenate(masks_all),
        'elevations': np.concatenate(elevs_all),
        'azimuths': np.concatenate(azs_all),
    }
    return metrics, details


def evaluate_wls(test_data):
    e3d = np.array([np.sqrt((e['delta_r']**2).sum()) for e in test_data])
    e2d = np.array([np.sqrt((e['delta_r'][:2]**2).sum()) for e in test_data])
    return {
        'rmse_3d': float(np.sqrt((e3d**2).mean())),
        'rmse_2d': float(np.sqrt((e2d**2).mean())),
        'median_3d': float(np.median(e3d)),
        'p95_3d': float(np.percentile(e3d, 95)),
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
        val_m, _ = evaluate(model, val_loader, device)
        if val_m['rmse_3d'] < best_val:
            best_val = val_m['rmse_3d']
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        if (ep+1) % 10 == 0:
            print(f"    {name} ep{ep+1:3d}: loss={loss:.4f} val_RMSE={val_m['rmse_3d']:.3f}m pat={patience_ctr}")
        if patience_ctr >= config.patience:
            print(f"    {name}: early stop ep{ep+1}")
            break

    model.load_state_dict(best_state)
    model.to(device)
    return model, n_params


# =====================================================================
# FIGURES
# =====================================================================
def generate_figures(model, test_loader, test_data, wls_m, cxtf_m, abl_results,
                     config, device, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    model.eval()
    cxtf_details_m, cxtf_details = evaluate(model, test_loader, device)

    # --- Figure: CDF ---
    wls_e3d = np.array([np.sqrt((e['delta_r']**2).sum()) for e in test_data])
    cxtf_e3d = cxtf_details['errors_3d']

    fig, ax = plt.subplots(figsize=(8, 5))
    for errs, label, color, ls in [
        (wls_e3d, f"WLS-SPP (median={np.median(wls_e3d):.1f}m)", 'red', '-'),
        (cxtf_e3d, f"CxTF (median={np.median(cxtf_e3d):.1f}m)", 'blue', '-'),
    ]:
        sorted_e = np.sort(errs)
        cdf = np.arange(1, len(sorted_e)+1) / len(sorted_e) * 100
        ax.plot(sorted_e, cdf, label=label, color=color, linestyle=ls, linewidth=2)
    ax.set_xlabel('3D Position Error (meters)')
    ax.set_ylabel('Cumulative Probability (%)')
    ax.set_title('Figure: Position Error CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(20, wls_e3d.max()))
    fig.savefig(os.path.join(output_dir, 'fig_cdf.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig_cdf.png")

    # --- Figure: Constellation Embeddings ---
    emb = model.const_emb.weight.detach().cpu().numpy()
    nc = emb.shape[0]
    # Cosine similarity
    cos_sim = np.zeros((nc, nc))
    for i in range(nc):
        for j in range(nc):
            cos_sim[i,j] = np.dot(emb[i], emb[j]) / (
                np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]) + 1e-8)

    names = [config.CONST_NAMES.get(i, f'C{i}') for i in range(nc)]
    colors_c = ['#2166AC', '#1B7837', '#B2182B', '#E66101']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # PCA
    if nc >= 2:
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(emb)
        for i in range(nc):
            ax1.scatter(emb_2d[i,0], emb_2d[i,1], c=colors_c[i], s=200,
                       marker='os^D'[i], zorder=5, edgecolors='white', linewidths=1.5)
            ax1.text(emb_2d[i,0]+0.1, emb_2d[i,1]+0.1, names[i],
                    fontsize=11, fontweight='bold', color=colors_c[i])
        ax1.set_title(f'PCA of Learned Embeddings\n(var: {pca.explained_variance_ratio_[0]:.0%}, {pca.explained_variance_ratio_[1]:.0%})')
        ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
        ax1.grid(True, alpha=0.3)

    # Cosine similarity heatmap
    im = ax2.imshow(cos_sim, cmap='RdYlGn', vmin=-1, vmax=1)
    for i in range(nc):
        for j in range(nc):
            ax2.text(j, i, f'{cos_sim[i,j]:.2f}', ha='center', va='center', fontsize=10)
    ax2.set_xticks(range(nc)); ax2.set_xticklabels(names)
    ax2.set_yticks(range(nc)); ax2.set_yticklabels(names)
    ax2.set_title('Pairwise Cosine Similarity')
    plt.colorbar(im, ax=ax2, shrink=0.8)
    fig.suptitle('Learned Constellation Embeddings', fontsize=14, fontweight='bold')
    fig.savefig(os.path.join(output_dir, 'fig_embeddings.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig_embeddings.png")

    # --- Figure: Selection by constellation and elevation ---
    scores = cxtf_details['scores']
    cids = cxtf_details['const_ids']
    masks = cxtf_details['masks']
    elevs = cxtf_details['elevations']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # By constellation
    const_scores = {}
    for c in range(nc):
        flat_s, flat_m, flat_c = [], [], []
        for b in range(scores.shape[0]):
            n = int(masks[b].sum())
            for i in range(n):
                if cids[b, i] == c:
                    flat_s.append(scores[b, i])
        if flat_s:
            const_scores[names[c]] = np.array(flat_s)

    positions = range(len(const_scores))
    ax1.bar(positions, [v.mean() for v in const_scores.values()],
            color=[colors_c[i] for i in range(len(const_scores))], alpha=0.8)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(const_scores.keys())
    ax1.set_ylabel('Mean Selection Score')
    ax1.set_title('Selection by Constellation')
    ax1.set_ylim(0, 1)

    # By elevation
    elev_bins = [(5, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90)]
    bin_scores = []
    bin_labels = []
    for lo, hi in elev_bins:
        flat_s = []
        for b in range(scores.shape[0]):
            n = int(masks[b].sum())
            for i in range(n):
                if lo <= elevs[b, i] < hi:
                    flat_s.append(scores[b, i])
        if flat_s:
            bin_scores.append(np.mean(flat_s))
            bin_labels.append(f'{lo}-{hi}')

    ax2.bar(range(len(bin_scores)), bin_scores, color='steelblue', alpha=0.8)
    ax2.set_xticks(range(len(bin_labels)))
    ax2.set_xticklabels(bin_labels, rotation=45)
    ax2.set_xlabel('Elevation (degrees)')
    ax2.set_ylabel('Mean Selection Score')
    ax2.set_title('Selection by Elevation')
    ax2.set_ylim(0, 1)

    fig.suptitle('Satellite Selection Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_selection.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved fig_selection.png")

    return cos_sim, names


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    config = Config()
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("CxTF FULL EXPERIMENT")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    print("=" * 70)

    # Load all traces
    print("\n[1/7] Loading data...")
    all_epochs = []
    for trace_dir in sorted(Path(args.data_dir).iterdir()):
        if trace_dir.is_dir():
            gp = trace_dir / 'device_gnss.csv'
            tp = trace_dir / 'ground_truth.csv'
            if gp.exists() and tp.exists():
                data = load_and_preprocess(str(gp), str(tp), config)
                all_epochs.extend(data)

    # Auto-detect constellations
    all_c = set()
    for e in all_epochs:
        all_c.update(e['const_ids'].tolist())
    config.n_constellations = max(all_c) + 1
    print(f"  Total: {len(all_epochs)} epochs, {len(all_c)} constellations")

    # Split
    n = len(all_epochs)
    n_train, n_val = int(0.70*n), int(0.15*n)
    train_data = all_epochs[:n_train]
    val_data = all_epochs[n_train:n_train+n_val]
    test_data = all_epochs[n_train+n_val:]
    print(f"  Split: train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    train_ds = GNSSDataset(train_data, config)
    stats = train_ds.get_stats()
    val_ds = GNSSDataset(val_data, config, stats=stats)
    test_ds = GNSSDataset(test_data, config, stats=stats)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, num_workers=2)

    # WLS Baseline
    print("\n[2/7] WLS Baseline...")
    wls_m = evaluate_wls(test_data)
    print(f"  WLS-SPP: 3D RMSE={wls_m['rmse_3d']:.2f}m, Median={wls_m['median_3d']:.2f}m, 95th={wls_m['p95_3d']:.2f}m")

    # Train CxTF
    print("\n[3/7] Training CxTF (full model)...")
    model, n_params = train_model(CxTF, config, train_loader, val_loader, device, "CxTF")
    print(f"  Parameters: {n_params:,}")

    cxtf_m, _ = evaluate(model, test_loader, device)
    imp = (1 - cxtf_m['rmse_3d'] / wls_m['rmse_3d']) * 100
    print(f"  CxTF: 3D RMSE={cxtf_m['rmse_3d']:.2f}m, Improv={imp:.1f}%")

    # Ablations
    print("\n[4/7] Ablation A: No constellation embeddings...")
    abl_a, _ = train_model(CxTF_NoEmb, config, train_loader, val_loader, device, "NoEmb")
    abl_a_m, _ = evaluate(abl_a, test_loader, device)

    print("\n[5/7] Ablation B: No satellite selection...")
    abl_b, _ = train_model(CxTF_NoSel, config, train_loader, val_loader, device, "NoSel")
    abl_b_m, _ = evaluate(abl_b, test_loader, device)

    print("\n[6/7] Ablation C: No elevation PE...")
    abl_c, _ = train_model(CxTF_NoPE, config, train_loader, val_loader, device, "NoPE")
    abl_c_m, _ = evaluate(abl_c, test_loader, device)

    abl_results = {'No Embedding': abl_a_m, 'No Selection': abl_b_m, 'No PE': abl_c_m}

    # Figures
    print("\n[7/7] Generating figures...")
    cos_sim, cnames = generate_figures(
        model, test_loader, test_data, wls_m, cxtf_m, abl_results,
        config, device, args.output_dir)

    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'cxtf_best.pt'))

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY (d_model={config.d_model}, L={config.n_layers}, params={n_params:,})")
    print(f"{'='*70}")
    hdr = f"{'Method':<28} {'3D RMSE':>8} {'2D RMSE':>8} {'Median':>8} {'95th':>8} {'Improv':>8}"
    print(hdr)
    print("-"*70)

    def row(nm, m, bl=None):
        ip = '—' if bl is None else f"{(1-m['rmse_3d']/bl)*100:.1f}%"
        return f"{nm:<28} {m['rmse_3d']:>7.2f}m {m['rmse_2d']:>7.2f}m {m['median_3d']:>7.2f}m {m['p95_3d']:>7.2f}m {ip:>8}"

    lines = []
    lines.append(row("WLS-SPP (baseline)", wls_m))
    lines.append(row("CxTF (proposed)", cxtf_m, wls_m['rmse_3d']))
    lines.append(row("Ablation A (no emb.)", abl_a_m, wls_m['rmse_3d']))
    lines.append(row("Ablation B (no sel.)", abl_b_m, wls_m['rmse_3d']))
    lines.append(row("Ablation C (no PE)", abl_c_m, wls_m['rmse_3d']))

    for l in lines:
        print(l)
    print("="*70)
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save text summary
    with open(os.path.join(args.output_dir, 'results_summary.txt'), 'w') as f:
        f.write(hdr + '\n' + '-'*70 + '\n')
        for l in lines:
            f.write(l + '\n')
        f.write('\nCosine Similarities:\n')
        for i in range(len(cnames)):
            for j in range(i+1, len(cnames)):
                f.write(f"  {cnames[i]}<->{cnames[j]}: {cos_sim[i,j]:.3f}\n")

    # Save JSON
    results = {
        'wls': wls_m, 'cxtf': cxtf_m, 'improvement_pct': imp,
        'ablation_a': abl_a_m, 'ablation_b': abl_b_m, 'ablation_c': abl_c_m,
        'cosine_similarity': cos_sim.tolist(),
        'config': {'d_model': config.d_model, 'n_layers': config.n_layers,
                   'n_heads': config.n_heads, 'params': n_params,
                   'n_constellations': config.n_constellations},
        'elapsed_seconds': elapsed,
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
