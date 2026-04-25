"""
Compute active-site ESM-2 embeddings for the Retroviral Wall pipeline.

This script extracts per-residue ESM-2 layer-33 representations, mean-pools
the YXDD +/- local catalytic window, and writes a cache consumed by pipeline.py.
The PCA compression is intentionally done inside each LOFO fold in pipeline.py
to avoid fitting representation transforms on held-out families.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def find_yxdd(seq):
    match = re.search(r"Y.DD", seq)
    return match.start() if match else None


def main():
    try:
        import esm
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'esm'. Install with: pip install fair-esm"
        ) from exc

    seq_df = pd.read_csv("data/rt_sequences.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    names = []
    embeddings = []
    with torch.no_grad():
        for _, row in seq_df.iterrows():
            name = row["rt_name"]
            seq = row["sequence"]
            anchor = find_yxdd(seq)
            if anchor is None:
                start, end = 0, min(len(seq), 24)
            else:
                start = max(0, anchor - 10)
                end = min(len(seq), anchor + 14)

            _, _, tokens = batch_converter([(name, seq)])
            tokens = tokens.to(device)
            out = model(tokens, repr_layers=[33], return_contacts=False)
            reps = out["representations"][33][0, 1 : len(seq) + 1].detach().cpu().numpy()
            pooled = reps[start:end].mean(axis=0)
            names.append(name)
            embeddings.append(pooled.astype(np.float32))
            print(f"{name:<25s} window={start}:{end} device={device}")

    out_path = Path("data/esm_active_site_embeddings.npz")
    np.savez_compressed(
        out_path,
        names=np.array(names, dtype=object),
        embeddings=np.vstack(embeddings).astype(np.float32),
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
