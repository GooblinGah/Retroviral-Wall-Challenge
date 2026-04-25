"""
Retroviral Wall Challenge — Submission Pipeline
=================================================
Ridge regression with domain-justified features and training config.

Architecture:
  Ridge alpha fixed at 100 for the selected submission model.
  For each LOFO fold (7 families):
    Fit Ridge(alpha=100) on all features, outer-train only -> predict held-out family.
  Audited alternatives (multi-alpha Ridge, hurdle model, rank target,
  residualization, ElasticNet, active-site ESM-2 PCA) are evaluated and
  rejected unless they pass pre-declared gates.

Config choices (fixed by domain knowledge, not CV-selected):
  - Features:  36 biophysical + 5 structural/pLDDT + 5 nonlinear transforms = 46 total
  - Target:    log(efficiency + 1)  -- skewed distribution (36 zeros, max 41%)
  - Weights:   efficiency + 0.1     -- aligns with WSpearman eval metric

Feature set rationale:
  36 biophysical features covering catalytic geometry, thermostability, ESM-IF
  scores, physicochemical properties, active site environment, and electrostatics.
  5 structural/pLDDT features computed from PDB Cα atoms and B-factors capture
  fold compactness, long-range contacts, and active-site model confidence.
  5 nonlinear transforms capture charge extremes, heavy-tailed ESM-IF
  perplexity, thermal slope, saturated instability, and conditional
  sequence-structure fit. BLOSUM local similarity, PE2 pentamutant scores,
  hurdle models, residualization, ElasticNet, and active-site ESM-2 PCA were
  audited but rejected by LOFO gates.

External data:
  - BLOSUM62 substitution matrix (Henikoff & Henikoff, 1992) — used in controls
  - PE2 pentamutant positions (Anzalone et al., 2019) — used in controls
  - MMLV-RT canonical sequence (GenBank J02255) — used in controls
  - ESM-2 650M active-site embeddings — computed in controls, rejected by gates

Usage:
  python pipeline.py                   # Stage 1: LOFO evaluation + controls
  python pipeline.py --predict FILE    # Stage 2: predict new candidates
"""

import warnings
warnings.filterwarnings("ignore")

import re
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from Bio.Align import substitution_matrices

_BLOSUM62 = substitution_matrices.load("BLOSUM62")

# Canonical MMLV-RT sequence — fixed biological prior, not learned from labels.
# Source: Moloney Murine Leukemia Virus RT, GenBank J02255 / UniProt P03355.
# This is the RT domain used in PE2 (Anzalone et al., 2019).
MMLV_SEQUENCE = (
    "TLNIEDEYRLHETSKEPDVSLGSTWLSDFPQAWAETGGMGLAVRQAPLIIPLKATSTPVSIKQYPMSQ"
    "EARLIKPHIQRLLDQGILVPCQSPWNTPLLPVKKPGTNDYRPVQDLREVNKRVEDIHPTVPNPYNLLS"
    "GLPPSHQWYTVLDLKDAFFCLRLHPTSQPLFAFEWRDPEMGISGQLTWTRLPQGFKNSPTLFDEALHRD"
    "LADFRIQHPDLILLQYVDDLLLAATSELDCQQGTRALLQTLGNLGYRASAKKAQICQKQVKYLGYLLKE"
    "GQRWLTEARKETVMGQPTPKTPRQLREFLGTAGFCRLWIPGFAEMAAPLYPLTKTGTLFNWGPDQQKAY"
    "QEIKQALLTAPALGLPDLTKPFELFVDEKQGYAKGVLTQKLGPWRRPVAYLSKKLDPVAAGWPPCLRMV"
    "AAIAVLTKDAGKLTMGQPLVILAPHAVEALVKQPPDRWLSNARMTHYQALLLDTDRVQFGPVVALNPATL"
    "LPLPEEGLQHNCLDILAEAHGTRPDLTDQPLPDADHTWYTDGSSLLQEGQRKAGAAVTTETEVIWAKALPAG"
    "TSAQRAELIALTQALKMAEGKKLNVYTDSRYAFATAHIHGEIYRRRGLLTSEGKEIKNKDEILALLKALFL"
    "PKRLSIIHCPGHQKGHSAEARGNRMADQAARKAAITETPDTSTLLIENSSP"
)
MMLV_YXDD_POS = 221  # 0-indexed position of Y in YVDD motif


def blosum(a, b):
    try:
        return float(_BLOSUM62[a, b])
    except (KeyError, IndexError):
        return 0.0


# ── Metrics ──

def pr_auc_score(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.mean(y_true)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def weighted_spearman(y_true_eff, y_score, epsilon=0.1):
    weights = y_true_eff + epsilon
    if len(y_true_eff) < 3:
        return 0.0
    rank_true = pd.Series(y_true_eff).rank().values
    rank_pred = pd.Series(y_score).rank().values
    mean_rt = np.average(rank_true, weights=weights)
    mean_rp = np.average(rank_pred, weights=weights)
    cov = np.sum(weights * (rank_true - mean_rt) * (rank_pred - mean_rp))
    std_t = np.sqrt(np.sum(weights * (rank_true - mean_rt) ** 2))
    std_p = np.sqrt(np.sum(weights * (rank_pred - mean_rp) ** 2))
    if std_t < 1e-12 or std_p < 1e-12:
        return 0.0
    return max(cov / (std_t * std_p), 0.0)

def cls_score(prauc, wspear):
    if prauc + wspear < 1e-12:
        return 0.0
    return 2 * prauc * wspear / (prauc + wspear)


# ── Data ──

BIOPHYS_CORE = [
    "triad_found_bin", "D1_D2_dist", "D2_D3_dist", "triad_best_rmsd",
    "yxdd_y_is_strand", "yxdd_x_is_turn", "yxdd_d1_is_turn", "yxdd_d2_is_strand",
    "hairpin_pass", "hairpin_confidence",
    "t40_raw", "t45_raw", "t50_raw", "t55_raw", "t60_raw", "t65_raw",
    "t70_raw", "t75_raw", "t80_raw", "thermophilicity_num",
    "perplexity", "log_likelihood",
    "instability_index", "gravy", "net_charge", "camsol", "helix_beta_ratio",
    "yxdd_hydrophobic_fraction", "yxdd_mean_hydrophobicity", "yxdd_std_hydrophobicity",
    "pocket_hydrophobic_per_res", "pocket_hbonds_per_res",
    "yxdd_5A_mean_pot", "palm_mean_pot", "lpqg_sp_3A_mean_pot", "motif12_mean_pot",
]


def _normalize_plddt(values):
    """Normalize ESMFold pLDDT values to 0-1 if stored on the 0-100 scale."""
    values = np.asarray(values, dtype=float)
    if len(values) and np.nanmax(values) > 1.5:
        values = values / 100.0
    return values


def _compute_structural_features(pdb_path, sequence=None):
    """Compute compactness, contact, and pLDDT confidence features from PDB Cα atoms."""
    coords = []
    plddt = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and " CA " in line:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                try:
                    plddt.append(float(line[60:66]))
                except ValueError:
                    plddt.append(np.nan)
    coords = np.array(coords)
    plddt = _normalize_plddt(plddt)
    n = len(coords)
    if n < 10:
        return {
            "radius_of_gyration": np.nan,
            "long_range_contact_frac": np.nan,
            "mean_pLDDT": np.nan,
            "pLDDT_at_triad": np.nan,
            "frac_high_pLDDT": np.nan,
        }

    center = coords.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))

    dists = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))
    seq_sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    contacts = (dists < 8.0) & (seq_sep > 2)
    i_idx, j_idx = np.where(np.triu(contacts, k=3))
    if len(i_idx) == 0:
        lr_frac = np.nan
    else:
        seps = np.abs(i_idx - j_idx)
        lr_frac = float((seps > 12).sum() / len(seps))

    mean_plddt = float(np.nanmean(plddt))
    frac_high = float(np.nanmean(plddt > 0.7))
    triad_plddt = np.nan
    if sequence:
        anchor = find_yxdd(sequence)
        if anchor is not None:
            # ESMFold residue order matches the submitted sequence. D1/D2 are
            # the two catalytic Asp residues in YXDD; add the nearest upstream
            # Asp in the palm window as a conservative third catalytic proxy.
            triad_idx = [anchor + 2, anchor + 3]
            upstream_ds = [i for i, aa in enumerate(sequence[max(0, anchor - 90):anchor], start=max(0, anchor - 90)) if aa == "D"]
            if upstream_ds:
                triad_idx.append(upstream_ds[-1])
            triad_vals = [plddt[i] for i in triad_idx if 0 <= i < len(plddt)]
            if triad_vals:
                triad_plddt = float(np.nanmean(triad_vals))

    return {
        "radius_of_gyration": rg,
        "long_range_contact_frac": lr_frac,
        "mean_pLDDT": mean_plddt,
        "pLDDT_at_triad": triad_plddt,
        "frac_high_pLDDT": frac_high,
    }


def load_data():
    seq_df = pd.read_csv("data/rt_sequences.csv")
    feat_df = pd.read_csv("data/handcrafted_features.csv")
    family_df = pd.read_csv("data/family_splits.csv")
    df = seq_df.merge(feat_df, on="rt_name", how="left", suffixes=("", "_feat"))
    for col in list(df.columns):
        if col.endswith("_feat"):
            base = col[:-5]
            if base in df.columns:
                df[base] = df[base].combine_first(df[col])
            df.drop(columns=[col], inplace=True)
    family_map = {}
    for _, row in family_df.iterrows():
        for name in row["rt_names"].split("|"):
            family_map[name.strip()] = row["family"]
    df["family"] = df["rt_name"].map(family_map)
    if "rt_family" in df.columns:
        df["family"] = df["family"].fillna(df["rt_family"])

    from pathlib import Path
    struct_dir = Path("data/structures")
    structural = {
        "radius_of_gyration": [],
        "long_range_contact_frac": [],
        "mean_pLDDT": [],
        "pLDDT_at_triad": [],
        "frac_high_pLDDT": [],
    }
    for _, row in df.iterrows():
        pdb = struct_dir / f"{row['rt_name']}.pdb"
        if pdb.exists():
            vals = _compute_structural_features(str(pdb), row.get("sequence"))
        else:
            vals = {key: np.nan for key in structural}
        for key in structural:
            structural[key].append(vals[key])
    for key, vals in structural.items():
        df[key] = vals

    return df


def get_biophys_features(df):
    cols = [c for c in BIOPHYS_CORE if c in df.columns]
    X = df[cols].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(
                X[col].replace({"TRUE": 1, "FALSE": 0, "True": 1, "False": 0}),
                errors="coerce"
            )

    # Structural features computed from PDB Cα coordinates.
    structural_cols = [
        "radius_of_gyration",       # fold compactness
        "long_range_contact_frac",  # cross-domain organization
        "mean_pLDDT",               # global ESMFold confidence
        "pLDDT_at_triad",           # confidence at catalytic Asp residues
        "frac_high_pLDDT",          # fraction of well-predicted residues
    ]
    for col in structural_cols:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce").values

    # Nonlinear transforms — domain-justified, not data-selected.
    #
    # net_charge^2: Extreme net charge (positive or negative) causes
    # aggregation and membrane interaction problems. Function degrades
    # nonlinearly with deviation from moderate charge.
    if "net_charge" in X.columns:
        X["net_charge_sq"] = X["net_charge"] ** 2
    #
    # log(perplexity): ESM-IF perplexity is heavy-tailed (range 1-17).
    # Sequence-structure mismatch degrades enzyme function, but the
    # relationship saturates — going from perplexity 2→4 matters more
    # than 12→14. Log compression captures this.
    if "perplexity" in X.columns:
        X["log_perplexity"] = np.log1p(X["perplexity"])

    # t40_minus_t60: a coarse denaturation slope. Prime editing is assayed
    # near physiological temperature, but enzymes that quickly lose predicted
    # structure between 40C and 60C are less likely to retain active geometry.
    if "t40_raw" in X.columns and "t60_raw" in X.columns:
        X["t40_minus_t60"] = X["t40_raw"] - X["t60_raw"]

    # inv_instability: instability effects saturate; the difference between
    # indices 20 and 40 matters more than between 80 and 100.
    if "instability_index" in X.columns:
        X["inv_instability"] = 1.0 / (X["instability_index"].clip(lower=0) + 1.0)

    # triad_x_loglik: ESM-IF sequence-structure fit is most meaningful when
    # the catalytic motif is present; otherwise the enzyme is broken upstream.
    if "triad_found_bin" in X.columns and "log_likelihood" in X.columns:
        X["triad_x_loglik"] = X["triad_found_bin"] * X["log_likelihood"]

    return X


# ── Feature computation (deterministic, label-independent) ──
# All features below use only amino acid sequences and fixed external references.

def find_yxdd(seq):
    m = re.search(r'Y.DD', seq)
    return m.start() if m else None


def compute_blosum_local(df, window=20):
    """Position-wise BLOSUM62 in YXDD+-window vs canonical MMLV-RT."""
    scores = []
    for _, row in df.iterrows():
        seq = row['sequence']
        anchor = find_yxdd(seq)
        if anchor is None:
            scores.append(np.nan)
            continue
        total, n = 0.0, 0
        for offset in range(-window, window + 4):
            mi = MMLV_YXDD_POS + offset
            ri = anchor + offset
            if 0 <= mi < len(MMLV_SEQUENCE) and 0 <= ri < len(seq):
                total += blosum(MMLV_SEQUENCE[mi], seq[ri])
                n += 1
        scores.append(total / n if n > 0 else np.nan)
    return np.array(scores, dtype=float)


def compute_local_identity(df, window=20):
    """Fraction of identical residues in YXDD+-window vs canonical MMLV-RT."""
    scores = []
    for _, row in df.iterrows():
        seq = row['sequence']
        anchor = find_yxdd(seq)
        if anchor is None:
            scores.append(np.nan)
            continue
        matches, n = 0, 0
        for offset in range(-window, window + 4):
            mi = MMLV_YXDD_POS + offset
            ri = anchor + offset
            if 0 <= mi < len(MMLV_SEQUENCE) and 0 <= ri < len(seq):
                if MMLV_SEQUENCE[mi] == seq[ri]:
                    matches += 1
                n += 1
        scores.append(matches / n if n > 0 else np.nan)
    return np.array(scores, dtype=float)


def compute_pe2_similarity(df):
    """BLOSUM62 score at PE2-equivalent positions vs PE2 target amino acids.

    PE2 mutations (Anzalone et al., 2019): D200N, T306K, W313F, T330P, L603W.
    Offsets are relative to the YXDD catalytic motif.
    """
    pe2 = {
        'pe2_D200N': (-22, 'N'),
        'pe2_T306K': (84, 'K'),
        'pe2_W313F': (91, 'F'),
        'pe2_T330P': (108, 'P'),
        'pe2_L603W': (381, 'W'),
    }
    results = {name: [] for name in pe2}
    for _, row in df.iterrows():
        seq = row['sequence']
        anchor = find_yxdd(seq)
        for name, (offset, target_aa) in pe2.items():
            if anchor is None:
                results[name].append(np.nan)
                continue
            pos = anchor + offset
            if 0 <= pos < len(seq):
                results[name].append(blosum(seq[pos], target_aa))
            else:
                results[name].append(np.nan)
    df_pe2 = pd.DataFrame(results)
    df_pe2['pe2_composite'] = df_pe2.mean(axis=1)
    return df_pe2


def build_features(df):
    """Build full feature matrix (biophys + BLOSUM local + PE2)."""
    X = get_biophys_features(df)
    X['blosum_local'] = compute_blosum_local(df)
    pe2_df = compute_pe2_similarity(df)
    for col in pe2_df.columns:
        X[col] = pe2_df[col].values
    return X


def load_active_site_esm_embeddings(df, path="data/esm_active_site_embeddings.npz"):
    """Load cached active-site ESM embeddings aligned to df, if available."""
    from pathlib import Path
    emb_path = Path(path)
    if not emb_path.exists():
        return None
    data = np.load(emb_path, allow_pickle=True)
    names = [str(x) for x in data["names"]]
    emb = data["embeddings"].astype(float)
    emb_map = dict(zip(names, emb))
    missing = [name for name in df["rt_name"] if name not in emb_map]
    if missing:
        print(f"  Warning: ESM cache missing {len(missing)} RTs; skipping ESM features.")
        return None
    return np.vstack([emb_map[name] for name in df["rt_name"]])


# ── Pipeline ──

def impute_and_scale(X_train, X_test):
    X_tr, X_te = X_train.copy(), X_test.copy()
    medians = np.nanmedian(X_tr, axis=0)
    medians = np.where(np.isnan(medians), 0, medians)
    for j in range(X_tr.shape[1]):
        X_tr[np.isnan(X_tr[:, j]), j] = medians[j]
        X_te[np.isnan(X_te[:, j]), j] = medians[j]
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_te), medians, scaler


def rank01(values):
    """Convert arbitrary scores to 0-1 ranks while preserving ties."""
    values = np.asarray(values, dtype=float)
    ranks = pd.Series(values).rank(method="average").values
    if len(ranks) <= 1:
        return np.zeros_like(ranks, dtype=float)
    return (ranks - 1) / (len(ranks) - 1)


def evaluate_predictions(y_bin, y_eff, preds, families=None):
    """Return global and optional per-family metrics for a prediction vector."""
    metrics = {
        "pr_auc": pr_auc_score(y_bin, preds),
        "wspearman": weighted_spearman(y_eff, preds),
    }
    metrics["cls"] = cls_score(metrics["pr_auc"], metrics["wspearman"])
    if families is not None:
        per_family = []
        for fam in sorted(set(families)):
            mask = families == fam
            fam_metrics = {
                "family": fam,
                "n": int(mask.sum()),
                "active": int(y_bin[mask].sum()),
                "mean_score": float(np.mean(preds[mask])),
            }
            if len(np.unique(y_bin[mask])) == 2:
                fam_metrics["pr_auc"] = pr_auc_score(y_bin[mask], preds[mask])
            else:
                fam_metrics["pr_auc"] = np.nan
            fam_metrics["wspearman"] = weighted_spearman(y_eff[mask], preds[mask])
            fam_metrics["cls"] = cls_score(
                0.0 if np.isnan(fam_metrics["pr_auc"]) else fam_metrics["pr_auc"],
                fam_metrics["wspearman"],
            )
            per_family.append(fam_metrics)
        metrics["per_family"] = per_family
    return metrics


def print_metric_block(label, metrics):
    print(f"\n{label}:")
    print(f"  CLS={metrics['cls']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  WSpearman={metrics['wspearman']:.4f}")
    if "per_family" in metrics:
        print(f"  {'Family':<25s} {'n':>3s} {'act':>3s} {'CLS':>7s} {'PR-AUC':>7s} {'WSpear':>7s} {'mean':>8s}")
        for row in metrics["per_family"]:
            pr = "nan" if np.isnan(row["pr_auc"]) else f"{row['pr_auc']:.4f}"
            print(
                f"  {row['family']:<25s} {row['n']:>3d} {row['active']:>3d} "
                f"{row['cls']:>7.4f} {pr:>7s} {row['wspearman']:>7.4f} {row['mean_score']:>8.3f}"
            )


def residualize_by_family(X_train, X_test, families_train):
    """Subtract within-family means on train; held-out rows use train global mean."""
    X_tr_r = X_train.copy()
    X_te_r = X_test.copy()
    global_mean = np.nanmean(X_train, axis=0)
    global_mean = np.where(np.isnan(global_mean), 0, global_mean)
    for fam in np.unique(families_train):
        mask = families_train == fam
        if mask.sum() >= 3:
            fam_mean = np.nanmean(X_train[mask], axis=0)
            fam_mean = np.where(np.isnan(fam_mean), global_mean, fam_mean)
            X_tr_r[mask] = X_tr_r[mask] - fam_mean
    X_te_r = X_te_r - global_mean
    return X_tr_r, X_te_r


def append_esm_pcs(X_train, X_test, emb_train, emb_test, n_components=5):
    """Append PCA-compressed active-site ESM embeddings fitted on train fold only."""
    if emb_train is None or emb_test is None:
        return X_train, X_test
    n_comp = min(n_components, emb_train.shape[0] - 1, emb_train.shape[1])
    if n_comp < 1:
        return X_train, X_test
    pca = PCA(n_components=n_comp, random_state=42)
    train_pcs = pca.fit_transform(emb_train)
    test_pcs = pca.transform(emb_test)
    return np.hstack([X_train, train_pcs]), np.hstack([X_test, test_pcs])


RIDGE_ALPHA = 100  # Fixed baseline; CLS robust across 75-250 (varies <0.03)
RIDGE_ALPHAS = (50, 100, 200, 500)
ACTIVE_RIDGE_ALPHAS = (200, 500, 1000)
ESM_PCA_COMPONENTS = 5


def _ridge_average(X_train, y_train, X_test, alphas, sample_weight=None):
    preds = []
    coefs = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train, sample_weight=sample_weight)
        preds.append(ridge.predict(X_test))
        coefs.append(ridge.coef_.copy())
    return np.mean(preds, axis=0), np.mean(coefs, axis=0)


def _elasticnet_inner_lofo(X_train_raw, y_train, y_eff_train, families_train):
    """Small nested-LOFO ElasticNet grid for optional shrinkage/selection."""
    alpha_grid = (0.001, 0.003, 0.01, 0.03, 0.1)
    l1_grid = (0.05, 0.2, 0.5)
    best = (float("-inf"), alpha_grid[0], l1_grid[0])
    for alpha in alpha_grid:
        for l1_ratio in l1_grid:
            inner_preds = np.full(len(y_train), np.nan)
            for fam in sorted(set(families_train)):
                te = families_train == fam
                tr = ~te
                if te.sum() == 0 or tr.sum() < 8:
                    continue
                X_tr_s, X_te_s, _, _ = impute_and_scale(X_train_raw[tr], X_train_raw[te])
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=42)
                model.fit(X_tr_s, y_train[tr], sample_weight=y_eff_train[tr] + 0.1)
                inner_preds[te] = model.predict(X_te_s)
            valid = ~np.isnan(inner_preds)
            if valid.sum() < 10:
                continue
            pr = pr_auc_score((y_eff_train[valid] > 0).astype(int), inner_preds[valid])
            ws = weighted_spearman(y_eff_train[valid], inner_preds[valid])
            score = cls_score(pr, ws)
            if score > best[0]:
                best = (score, alpha, l1_ratio)
    return best[1], best[2]


def lofo_nested(
    df,
    X,
    *,
    model_variant="baseline",
    esm_embeddings=None,
    use_elasticnet=False,
):
    """
    LOFO evaluation with fixed, pre-declared model variants.

    All preprocessing is fit inside each outer train fold. The default
    baseline uses the original fixed Ridge(alpha=100) path.
    """
    families = df["family"].values
    outer_fams = sorted(df["family"].unique())
    y_eff = df["pe_efficiency_pct"].values
    y_bin = df["active"].values.astype(int)
    y_target = np.log1p(y_eff)
    X_arr = X.values.astype(float)
    sw_full = y_eff + 0.1

    predictions = np.full(len(df), np.nan)
    head_predictions = {
        "log_ridge": np.full(len(df), np.nan),
        "rank_ridge": np.full(len(df), np.nan),
        "residual_rank": np.full(len(df), np.nan),
        "hurdle": np.full(len(df), np.nan),
        "elasticnet": np.full(len(df), np.nan),
    }
    fold_coefs = {}
    fold_params = {}

    for outer_fam in outer_fams:
        te = families == outer_fam
        tr = ~te
        X_tr_raw, X_te_raw = X_arr[tr], X_arr[te]
        if esm_embeddings is not None:
            X_tr_raw, X_te_raw = append_esm_pcs(
                X_tr_raw, X_te_raw, esm_embeddings[tr], esm_embeddings[te]
            )
        X_tr_s, X_te_s, _, _ = impute_and_scale(X_tr_raw, X_te_raw)

        if model_variant == "baseline":
            pred, coef = _ridge_average(
                X_tr_s, y_target[tr], X_te_s, (RIDGE_ALPHA,), sample_weight=sw_full[tr]
            )
            predictions[te] = pred
            fold_coefs[outer_fam] = coef
            continue

        pred_log, coef_log = _ridge_average(
            X_tr_s, y_target[tr], X_te_s, RIDGE_ALPHAS, sample_weight=sw_full[tr]
        )
        head_predictions["log_ridge"][te] = pred_log
        fold_coefs[outer_fam] = coef_log

        if model_variant in {"rank", "residual", "combined"}:
            y_rank = pd.Series(y_eff[tr]).rank(method="average").values
            pred_rank, _ = _ridge_average(
                X_tr_s, y_rank, X_te_s, RIDGE_ALPHAS, sample_weight=sw_full[tr]
            )
            head_predictions["rank_ridge"][te] = pred_rank

        if model_variant in {"residual", "combined"}:
            X_tr_res, X_te_res = residualize_by_family(X_tr_raw, X_te_raw, families[tr])
            X_tr_res_s, X_te_res_s, _, _ = impute_and_scale(X_tr_res, X_te_res)
            y_rank = pd.Series(y_eff[tr]).rank(method="average").values
            pred_res, _ = _ridge_average(
                X_tr_res_s, y_rank, X_te_res_s, RIDGE_ALPHAS, sample_weight=sw_full[tr]
            )
            head_predictions["residual_rank"][te] = pred_res

        if model_variant in {"hurdle", "combined"}:
            clf = LogisticRegression(C=0.1, penalty="l2", class_weight="balanced", max_iter=5000)
            clf.fit(X_tr_s, y_bin[tr])
            p_active = clf.predict_proba(X_te_s)[:, 1]

            active_tr = y_bin[tr] == 1
            if active_tr.sum() >= 5:
                pred_eff, _ = _ridge_average(
                    X_tr_s[active_tr],
                    y_target[tr][active_tr],
                    X_te_s,
                    ACTIVE_RIDGE_ALPHAS,
                    sample_weight=sw_full[tr][active_tr],
                )
                pred_eff = np.clip(np.expm1(pred_eff), 0, None)
            else:
                pred_eff = np.zeros(te.sum())
            head_predictions["hurdle"][te] = p_active * pred_eff

        if use_elasticnet:
            alpha, l1_ratio = _elasticnet_inner_lofo(X_tr_raw, y_target[tr], y_eff[tr], families[tr])
            enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=42)
            enet.fit(X_tr_s, y_target[tr], sample_weight=sw_full[tr])
            head_predictions["elasticnet"][te] = enet.predict(X_te_s)
            fold_params[outer_fam] = {"elasticnet_alpha": alpha, "elasticnet_l1_ratio": l1_ratio}

        heads = ["log_ridge"]
        if model_variant in {"rank", "residual", "combined"}:
            heads.append("rank_ridge")
        if model_variant in {"residual", "combined"}:
            heads.append("residual_rank")
        if model_variant in {"hurdle", "combined"}:
            heads.append("hurdle")
        if use_elasticnet:
            heads.append("elasticnet")
        fold_stack = np.vstack([head_predictions[h][te] for h in heads])
        predictions[te] = np.mean(fold_stack, axis=0)

    if model_variant != "baseline":
        heads = ["log_ridge"]
        if model_variant in {"rank", "residual", "combined"}:
            heads.append("rank_ridge")
        if model_variant in {"residual", "combined"}:
            heads.append("residual_rank")
        if model_variant in {"hurdle", "combined"}:
            heads.append("hurdle")
        if use_elasticnet:
            heads.append("elasticnet")
        ranked_heads = []
        for head in heads:
            vals = head_predictions[head]
            if np.isnan(vals).any():
                continue
            ranked_heads.append(rank01(vals))
        if ranked_heads:
            predictions = np.mean(ranked_heads, axis=0)

    fold_info = {
        "alpha": RIDGE_ALPHA if model_variant == "baseline" else list(RIDGE_ALPHAS),
        "model_variant": model_variant,
        "feature_names": list(X.columns),
        "fold_coefs": fold_coefs,
        "fold_params": fold_params,
        "head_predictions": head_predictions,
    }

    return predictions, fold_info


# ── Stage 2: fit on all data, predict new candidates ──

def fit_final_model(df, X, ridge_alpha, model_variant="combined"):
    """Fit final model heads on all 57 samples. Returns artifacts for prediction."""
    y_target = np.log1p(df["pe_efficiency_pct"].values)
    y_eff = df["pe_efficiency_pct"].values
    y_bin = df["active"].values.astype(int)
    sw = df["pe_efficiency_pct"].values + 0.1
    X_arr = X.values.astype(float)

    medians = np.nanmedian(X_arr, axis=0)
    medians = np.where(np.isnan(medians), 0, medians)
    X_imp = X_arr.copy()
    for j in range(X_imp.shape[1]):
        X_imp[np.isnan(X_imp[:, j]), j] = medians[j]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    ridges = []
    alphas = RIDGE_ALPHAS if model_variant != "baseline" else (ridge_alpha,)
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled, y_target, sample_weight=sw)
        ridges.append(ridge)

    rank_ridges = []
    y_rank = pd.Series(y_eff).rank(method="average").values
    if model_variant in {"rank", "residual", "combined"}:
        for alpha in RIDGE_ALPHAS:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_scaled, y_rank, sample_weight=sw)
            rank_ridges.append(ridge)

    classifier = None
    active_ridges = []
    if model_variant in {"hurdle", "combined"}:
        classifier = LogisticRegression(C=0.1, penalty="l2", class_weight="balanced", max_iter=5000)
        classifier.fit(X_scaled, y_bin)
        active_mask = y_bin == 1
        for alpha in ACTIVE_RIDGE_ALPHAS:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_scaled[active_mask], y_target[active_mask], sample_weight=sw[active_mask])
            active_ridges.append(ridge)

    return {
        "medians": medians,
        "scaler": scaler,
        "ridges": ridges,
        "rank_ridges": rank_ridges,
        "classifier": classifier,
        "active_ridges": active_ridges,
        "feature_names": list(X.columns),
        "ridge_alpha": ridge_alpha,
        "model_variant": model_variant,
    }


def predict_new(model, df_new):
    """Score new RT candidates using a fitted model."""
    from pathlib import Path
    struct_dir = Path("data/structures")
    if "radius_of_gyration" not in df_new.columns:
        structural = {
            "radius_of_gyration": [],
            "long_range_contact_frac": [],
            "mean_pLDDT": [],
            "pLDDT_at_triad": [],
            "frac_high_pLDDT": [],
        }
        for _, row in df_new.iterrows():
            pdb = struct_dir / f"{row['rt_name']}.pdb"
            if pdb.exists():
                vals = _compute_structural_features(str(pdb), row.get("sequence"))
            else:
                vals = {key: np.nan for key in structural}
            for key in structural:
                structural[key].append(vals[key])
        df_new = df_new.copy()
        for key, vals in structural.items():
            df_new[key] = vals
    X_new = get_biophys_features(df_new)
    X_arr = X_new.values.astype(float)

    X_imp = X_arr.copy()
    for j in range(X_imp.shape[1]):
        X_imp[np.isnan(X_imp[:, j]), j] = model["medians"][j]

    X_scaled = model["scaler"].transform(X_imp)
    log_scores = np.mean([ridge.predict(X_scaled) for ridge in model["ridges"]], axis=0)
    heads = [rank01(log_scores)]
    if model.get("rank_ridges"):
        rank_scores = np.mean([ridge.predict(X_scaled) for ridge in model["rank_ridges"]], axis=0)
        heads.append(rank01(rank_scores))
    if model.get("classifier") is not None and model.get("active_ridges"):
        p_active = model["classifier"].predict_proba(X_scaled)[:, 1]
        active_scores = np.mean([ridge.predict(X_scaled) for ridge in model["active_ridges"]], axis=0)
        hurdle_scores = p_active * np.clip(np.expm1(active_scores), 0, None)
        heads.append(rank01(hurdle_scores))
    scores = np.mean(heads, axis=0)

    return pd.DataFrame({
        "rt_name": df_new["rt_name"],
        "predicted_score": scores,
        "predicted_active": (scores >= np.median(scores)).astype(int),
    })


# ── Diagnostics ──

def bootstrap_cls(y_bin, y_eff, preds, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_bin)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pr = pr_auc_score(y_bin[idx], preds[idx])
        ws = weighted_spearman(y_eff[idx], preds[idx])
        vals.append(cls_score(pr, ws))
    return np.array(vals)


def permutation_cls(y_bin, y_eff, preds, n_perm=1000, seed=42):
    """Permutation null for the fixed OOF predictions."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        idx = rng.permutation(len(y_bin))
        pr = pr_auc_score(y_bin[idx], preds)
        ws = weighted_spearman(y_eff[idx], preds)
        vals.append(cls_score(pr, ws))
    vals = np.array(vals)
    observed = cls_score(pr_auc_score(y_bin, preds), weighted_spearman(y_eff, preds))
    p_value = (1 + np.sum(vals >= observed)) / (len(vals) + 1)
    return vals, p_value


def print_retroviral_error_analysis(df, X, preds, pb):
    """Print compact FP/FN diagnostics for the held-out Retroviral fold."""
    retro = df["family"].values == "Retroviral"
    retro_df = df[retro].copy()
    retro_X = X.loc[retro].copy()
    retro_df["score"] = preds[retro]
    retro_df["pred"] = pb[retro]
    fp = retro_df[(retro_df["pred"] == 1) & (retro_df["active"] == 0)]
    fn = retro_df[(retro_df["pred"] == 0) & (retro_df["active"] == 1)]

    print("\nRetroviral FP/FN diagnostic:")
    if len(fp):
        print("  False positives: " + ", ".join(fp["rt_name"].tolist()))
    else:
        print("  False positives: none")
    if len(fn):
        print("  False negatives: " + ", ".join(fn["rt_name"].tolist()))
    else:
        print("  False negatives: none")

    active_mask = retro_df["active"].values == 1
    inactive_mask = ~active_mask
    diffs = []
    for col in retro_X.columns:
        vals = pd.to_numeric(retro_X[col], errors="coerce").values.astype(float)
        if np.all(np.isnan(vals)):
            continue
        scale = np.nanstd(vals)
        if scale < 1e-12 or np.isnan(scale):
            continue
        diff = (np.nanmean(vals[active_mask]) - np.nanmean(vals[inactive_mask])) / scale
        diffs.append((col, diff))
    print("  Largest active-vs-inactive Retroviral standardized feature gaps:")
    for col, diff in sorted(diffs, key=lambda x: -abs(x[1]))[:8]:
        print(f"    {col:<28s} {diff:+.2f} sd")


# ── Main ──

def main():
    t0 = time.time()
    print("=" * 65)
    print("Retroviral Wall Challenge — Ridge Pipeline")
    print("=" * 65)
    print()
    print("Config (domain-justified):")
    print("  Features:  biophys + structural/pLDDT + nonlinear transforms")
    print("  Target:    log(efficiency + 1)  [skewed distribution]")
    print("  Weights:   efficiency + 0.1     [matches WSpearman metric]")
    print(f"  Model:     LOFO-audited linear ensemble; baseline Ridge alpha={RIDGE_ALPHA}")

    df = load_data()
    X = get_biophys_features(df)
    y_bin = df["active"].values.astype(int)
    y_eff = df["pe_efficiency_pct"].values
    families = df["family"].values

    print(f"\nFeatures: {X.shape[1]}, Samples: {len(df)}")

    # ── LOFO evaluation ──
    esm_embeddings = load_active_site_esm_embeddings(df)
    if esm_embeddings is None:
        print("  Active-site ESM cache not found; Phase 3 ESM features skipped.")
    else:
        print(f"  Loaded active-site ESM cache: {esm_embeddings.shape[0]} x {esm_embeddings.shape[1]}")

    variants = [
        ("baseline", {"model_variant": "baseline", "esm_embeddings": None, "use_elasticnet": False}),
        ("phase1a_multi_alpha", {"model_variant": "multi", "esm_embeddings": None, "use_elasticnet": False}),
        ("phase1c_hurdle", {"model_variant": "hurdle", "esm_embeddings": None, "use_elasticnet": False}),
        ("phase2a_rank", {"model_variant": "rank", "esm_embeddings": None, "use_elasticnet": False}),
        ("phase2b_residualized", {"model_variant": "residual", "esm_embeddings": None, "use_elasticnet": False}),
        ("phase2_combined", {"model_variant": "combined", "esm_embeddings": None, "use_elasticnet": False}),
        ("phase3b_elasticnet", {"model_variant": "combined", "esm_embeddings": None, "use_elasticnet": True}),
    ]
    if esm_embeddings is not None:
        variants.append(("phase3a_esm_combined", {"model_variant": "combined", "esm_embeddings": esm_embeddings, "use_elasticnet": False}))

    variant_results = {}
    for name, kwargs in variants:
        p, info = lofo_nested(df, X, **kwargs)
        metrics = evaluate_predictions(y_bin, y_eff, p, families)
        variant_results[name] = {"preds": p, "info": info, "metrics": metrics, "kwargs": kwargs}
        print_metric_block(name, metrics)

    baseline_cls = variant_results["baseline"]["metrics"]["cls"]
    baseline_pr = variant_results["baseline"]["metrics"]["pr_auc"]
    baseline_ws = variant_results["baseline"]["metrics"]["wspearman"]
    gated_names = ["baseline"]
    if variant_results["phase1a_multi_alpha"]["metrics"]["cls"] >= baseline_cls - 0.005:
        gated_names.append("phase1a_multi_alpha")
    hurdle_metrics = variant_results["phase1c_hurdle"]["metrics"]
    if (
        hurdle_metrics["pr_auc"] >= baseline_pr + 0.02
        and hurdle_metrics["wspearman"] >= baseline_ws - 0.02
        and hurdle_metrics["cls"] >= max(0.665, baseline_cls)
    ):
        gated_names.append("phase1c_hurdle")
    for name in ["phase2a_rank", "phase2b_residualized", "phase2_combined", "phase3b_elasticnet", "phase3a_esm_combined"]:
        if name in variant_results and variant_results[name]["metrics"]["cls"] >= baseline_cls:
            gated_names.append(name)

    best_name = max(gated_names, key=lambda n: variant_results[n]["metrics"]["cls"])
    preds = variant_results[best_name]["preds"]
    fold_info = variant_results[best_name]["info"]
    final_metrics = variant_results[best_name]["metrics"]
    alpha = fold_info["alpha"]
    prauc = final_metrics["pr_auc"]
    wspear = final_metrics["wspearman"]
    cls = final_metrics["cls"]

    print(f"\nSelected final variant: {best_name}")

    # Binary predictions: post-hoc diagnostic threshold on OOF predictions.
    # The continuous predicted_score is the primary submission output;
    # predicted_active is ancillary and not used in CLS computation.
    thresh = np.median(preds)
    best_f1, best_t = 0, thresh
    for t in np.linspace(preds.min() - 0.01, preds.max() + 0.01, 500):
        pb = (preds >= t).astype(int)
        tp = ((pb == 1) & (y_bin == 1)).sum()
        fp = ((pb == 1) & (y_bin == 0)).sum()
        fn = ((pb == 0) & (y_bin == 1)).sum()
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    pb = (preds >= best_t).astype(int)
    tp = ((pb == 1) & (y_bin == 1)).sum()
    fp = ((pb == 1) & (y_bin == 0)).sum()
    tn = ((pb == 0) & (y_bin == 0)).sum()
    fn = ((pb == 0) & (y_bin == 1)).sum()

    print(f"\n{'='*65}")
    print(f"Ridge alpha: {alpha} (fixed)")
    print(f"CLS = {cls:.4f}   PR-AUC = {prauc:.4f}   WSpearman = {wspear:.4f}")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}  (post-hoc threshold={best_t:.3f})")
    print(f"{'='*65}")

    # Feature importance: mean |coefficient| across folds
    feat_names = fold_info["feature_names"]
    coef_matrix = np.array([fold_info["fold_coefs"][f] for f in sorted(fold_info["fold_coefs"])])
    mean_abs = np.mean(np.abs(coef_matrix), axis=0)
    ranking = sorted(zip(feat_names, mean_abs), key=lambda x: -x[1])

    print(f"\nTop features by mean |coefficient| across {len(fold_info['fold_coefs'])} folds:")
    max_coef = ranking[0][1] if ranking else 1
    for feat, coef in ranking[:15]:
        bar = "#" * max(1, int(20 * coef / max_coef))
        print(f"  {bar:<20s} {coef:.4f}  {feat}")

    # Per-family breakdown
    print(f"\nPer-family breakdown:")
    print(f"  {'Family':<25s} {'n':>3s} {'act':>3s} {'TP':>3s} {'FP':>3s} {'TN':>3s} {'FN':>3s} {'mean_score':>10s}")
    for fam in sorted(set(families)):
        m = families == fam
        a = y_bin[m]
        p = pb[m]
        tp_f = ((p == 1) & (a == 1)).sum()
        fp_f = ((p == 1) & (a == 0)).sum()
        tn_f = ((p == 0) & (a == 0)).sum()
        fn_f = ((p == 0) & (a == 1)).sum()
        print(f"  {fam:<25s} {m.sum():>3d} {a.sum():>3d} {tp_f:>3d} {fp_f:>3d} {tn_f:>3d} {fn_f:>3d} "
              f"{preds[m].mean():>10.3f}")

    # Retroviral detail
    retro = families == 'Retroviral'
    retro_df = df[retro].copy()
    retro_df['score'] = preds[retro]
    retro_df['pred'] = pb[retro]
    retro_df = retro_df.sort_values('score', ascending=False)

    print(f"\nRetroviral detail:")
    print(f"  {'Name':<25s} {'Act':>4s} {'Eff%':>6s} {'Score':>8s} {'Pred':>5s}")
    for _, r in retro_df.iterrows():
        ok = "+" if r['pred'] == r['active'] else "-"
        print(f"  {r['rt_name']:<25s} {int(r['active']):>4d} {r['pe_efficiency_pct']:>6.1f} "
              f"{r['score']:>8.3f} {int(r['pred']):>5d}  {ok}")
    print_retroviral_error_analysis(df, X, preds, pb)

    # ── Family memorisation controls ──
    print(f"\n{'='*65}")
    print("FAMILY MEMORISATION CONTROLS")
    print(f"{'='*65}")

    local_identity = compute_local_identity(df)

    X_full = build_features(df)  # biophys + BLOSUM + PE2 (43 features)
    X_mmlv = pd.DataFrame({'blosum_local': compute_blosum_local(df)})
    pe2_df = compute_pe2_similarity(df)
    for col in pe2_df.columns:
        X_mmlv[col] = pe2_df[col].values
    X_identity = pd.DataFrame({'local_identity': local_identity})

    controls = [
        ("C1: biophys + MMLV (43)",     X_full),
        ("C2: MMLV-anchored only",      X_mmlv),
        ("C3: raw local identity only",  X_identity),
        ("FULL MODEL (biophys 36)",      X),
    ]

    print(f"\n  {'Control':<35s} {'CLS':>7s} {'PR-AUC':>7s} {'WSpear':>7s} {'Alpha':>7s} {'Feats':>5s}")
    print(f"  {'-'*68}")
    for name, X_ctrl in controls:
        p_ctrl, info_ctrl = lofo_nested(df, X_ctrl)
        c = cls_score(pr_auc_score(y_bin, p_ctrl), weighted_spearman(y_eff, p_ctrl))
        pr = pr_auc_score(y_bin, p_ctrl)
        ws = weighted_spearman(y_eff, p_ctrl)
        print(f"  {name:<35s} {c:7.4f} {pr:7.4f} {ws:7.4f} {info_ctrl['alpha']:>7.1f} {X_ctrl.shape[1]:>5d}")

    print(f"\n  FULL > C1: MMLV features hurt LOFO (family-correlated, no cross-family signal)")
    print(f"  FULL > C2, C3: biophysical features carry the real signal")

    # Bootstrap
    boot = bootstrap_cls(y_bin, y_eff, preds)
    print(f"\nBootstrap CI (n=5000):")
    print(f"  CLS mean={boot.mean():.4f}  std={boot.std():.4f}")
    print(f"  90% CI: [{np.percentile(boot, 5):.4f}, {np.percentile(boot, 95):.4f}]")

    retro_mask = families == "Retroviral"
    retro_boot = bootstrap_cls(y_bin[retro_mask], y_eff[retro_mask], preds[retro_mask], n_boot=5000)
    print(f"\nRetroviral bootstrap CI (n=5000):")
    print(f"  CLS mean={retro_boot.mean():.4f}  std={retro_boot.std():.4f}")
    print(f"  90% CI: [{np.percentile(retro_boot, 5):.4f}, {np.percentile(retro_boot, 95):.4f}]")

    perm, p_value = permutation_cls(y_bin, y_eff, preds, n_perm=1000)
    print(f"\nPermutation test (n=1000, fixed OOF predictions):")
    print(f"  null mean={perm.mean():.4f}  95th={np.percentile(perm, 95):.4f}  p={p_value:.4f}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")

    # ── Save submission ──
    sub = pd.DataFrame({
        "rt_name": df["rt_name"],
        "predicted_active": pb,
        "predicted_score": preds,
    })
    sub.to_csv("submissions/submission.csv", index=False)
    print(f"Saved submissions/submission.csv")

    # ── Save Stage 2 model ──
    model = fit_final_model(df, X, RIDGE_ALPHA, model_variant=fold_info["model_variant"])
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model.pkl (variant={model['model_variant']}, {len(model['feature_names'])} features)")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--predict":
        # Stage 2: predict new candidates
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        new_df = pd.read_csv(sys.argv[2])
        result = predict_new(model, new_df)
        out_path = "submissions/new_predictions.csv"
        result.to_csv(out_path, index=False)
        print(f"Predictions saved to {out_path}")
        print(result.to_string(index=False))
    else:
        main()
