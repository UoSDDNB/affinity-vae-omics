#!/usr/bin/env python3
"""
Script to generate synthetic scRNA-seq gene-expression profiles using a trained VAE decoder.

This script:
1. Loads a trained DecoderC from a checkpoint
2. Samples latent points from real cell latent distributions
3. Decodes latent points into synthetic gene-expression profiles
4. Inserts synthetic cells into an existing AnnData object
5. Recomputes UMAP embeddings with the synthetic cells
6. Saves a combined .h5ad (real + synthetic cells) compatible with Dataset_h5ad_reader
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import umap
import matplotlib.pyplot as plt
def _get_latent_columns(meta_df: pd.DataFrame) -> list[str]:
    """Return latent mean column names sorted by their index."""
    lat_cols = [
        col for col in meta_df.columns if col.startswith("lat") and col[3:].isdigit()
    ]
    return sorted(lat_cols, key=lambda c: int(c[3:]))


def _get_std_columns(meta_df: pd.DataFrame) -> list[str]:
    """Return latent std column names sorted by their index."""
    std_cols = [
        col for col in meta_df.columns if col.startswith("std-") and col[4:].isdigit()
    ]
    return sorted(std_cols, key=lambda c: int(c[4:]))



# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from avae.decoders.decoders import DecoderC

# ============================================================================
# Functions
# ============================================================================


def load_decoder(checkpoint_path: str, device: torch.device = torch.device("cpu")) -> DecoderC:
    """
    Load DecoderC from a checkpoint file.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file (.pt)
    device : torch.device
        Device to load the model on (default: CPU)
        
    Returns
    -------
    decoder : DecoderC
        Loaded decoder model in eval mode
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to extract decoder from full model
    if "model_class_object" in checkpoint:
        print("Extracting decoder from full model...")
        full_model = checkpoint["model_class_object"]
        
        # Load full model state dict if available
        if "model_state_dict" in checkpoint:
            print("Loading full model state dict...")
            full_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        # Check if model has decoder attribute
        if hasattr(full_model, "decoder"):
            decoder = full_model.decoder
        elif hasattr(full_model, "dec"):
            decoder = full_model.dec
        else:
            raise ValueError("Could not find decoder in model. Model attributes: " + 
                           str([attr for attr in dir(full_model) if not attr.startswith('_')]))
    else:
        raise ValueError("Checkpoint does not contain 'model_class_object'. " +
                        "Available keys: " + str(checkpoint.keys()))
    
    decoder.to(device)
    decoder.eval()
    
    print("Decoder loaded successfully.")
    return decoder


def load_latent_metadata(meta_path: str) -> pd.DataFrame:
    """
    Load latent metadata from pickle file.
    
    Parameters
    ----------
    meta_path : str
        Path to the metadata pickle file
        
    Returns
    -------
    meta_df : pd.DataFrame
        DataFrame containing latent means (lat0-latN) and stds (std-0 to std-N)
    """
    print(f"Loading latent metadata from: {meta_path}")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    meta_df = pd.read_pickle(meta_path)
    print(f"Loaded metadata with {len(meta_df)} cells")
    
    return meta_df


def infer_latent_dims(meta_df: pd.DataFrame) -> int:
    """
    Infer latent dimensionality from metadata DataFrame.
    
    Parameters
    ----------
    meta_df : pd.DataFrame
        DataFrame with columns lat0, lat1, ... and std-0, std-1, ...
        
    Returns
    -------
    latent_dims : int
        Number of latent dimensions
    """
    # Find all columns starting with 'lat' and extract numbers
    lat_cols = [col for col in meta_df.columns if col.startswith("lat") and col[3:].isdigit()]
    
    if not lat_cols:
        raise ValueError("No latent dimension columns found (expected 'lat0', 'lat1', ...)")
    
    # Extract the maximum index
    max_idx = max(int(col[3:]) for col in lat_cols)
    latent_dims = max_idx + 1
    
    print(f"Inferred latent dimensionality: {latent_dims}")
    return latent_dims


def normalize_celltype_label(label: str) -> str:
    """Mirror Dataset_h5ad_reader label normalization."""
    return str(label).replace(", ", "--").replace(" ", "-")


def resolve_h5ad_row(meta_df: pd.DataFrame, cell_idx: int, n_obs: int) -> int:
    """
    Map a meta-pickle row index to the corresponding h5ad row.

    Training meta stores the h5ad row in the ``filename`` column.
    """
    if cell_idx >= len(meta_df):
        raise ValueError(f"Cell index {cell_idx} out of range (max: {len(meta_df) - 1})")

    if "filename" not in meta_df.columns:
        raise ValueError(
            "Metadata missing 'filename' column; cannot map meta row to h5ad row."
        )

    row = int(meta_df.iloc[cell_idx]["filename"])
    if row < 0 or row >= n_obs:
        raise ValueError(
            f"Meta row {cell_idx} maps to h5ad row {row}, "
            f"but adata has {n_obs} cells."
        )
    return row


def get_source_celltype(
    adata: ad.AnnData,
    meta_df: pd.DataFrame,
    cell_idx: int,
    column_name: str,
) -> str:
    """Return the normalized cell type of the source real cell."""
    if column_name not in adata.obs.columns:
        raise ValueError(
            f"Column '{column_name}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    h5ad_row = resolve_h5ad_row(meta_df, cell_idx, adata.n_obs)
    raw_label = adata.obs.iloc[h5ad_row][column_name]
    return normalize_celltype_label(raw_label)


def finalize_combined_adata(
    adata: ad.AnnData,
    cell_type_column_name: str,
) -> ad.AnnData:
    """
    Ensure obs columns required for VAE h5ad loading and synthetic filtering.
    """
    if cell_type_column_name not in adata.obs.columns:
        raise ValueError(
            f"Column '{cell_type_column_name}' missing from combined AnnData.obs."
        )

    if "origin" not in adata.obs.columns:
        adata.obs["origin"] = "real"
    else:
        adata.obs["origin"] = adata.obs["origin"].fillna("real")

    if "is_synthetic" not in adata.obs.columns:
        adata.obs["is_synthetic"] = adata.obs["origin"] != "real"
    else:
        adata.obs["is_synthetic"] = adata.obs["is_synthetic"].fillna(
            adata.obs["origin"] != "real"
        )

    return adata


def validate_h5ad_output(
    output_path: str,
    cell_type_column_name: str,
    expected_n_obs: int,
    expected_n_vars: int,
) -> None:
    """Smoke-check that the written h5ad is readable and has expected shape."""
    check = ad.read_h5ad(output_path, backed="r")
    try:
        if cell_type_column_name not in check.obs.columns:
            raise ValueError(
                f"Written h5ad missing '{cell_type_column_name}' in obs."
            )
        if check.n_obs != expected_n_obs:
            raise ValueError(
                f"Expected {expected_n_obs} cells, found {check.n_obs}."
            )
        if check.n_vars != expected_n_vars:
            raise ValueError(
                f"Expected {expected_n_vars} genes, found {check.n_vars}."
            )
        print(
            f"h5ad validation passed: {check.n_obs} cells, {check.n_vars} genes."
        )
    finally:
        if hasattr(check, "file") and check.file is not None:
            check.file.close()


def _get_latent_stats(
    meta_df: pd.DataFrame, cell_idx: int, latent_dims: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract latent means and standard deviations for a given cell.

    Parameters
    ----------
    meta_df : pd.DataFrame
        DataFrame containing latent statistics
    cell_idx : int
        Index of the real cell to sample from
    latent_dims : int
        Number of latent dimensions

    Returns
    -------
    means : np.ndarray
        Mean vector of shape (latent_dims,)
    stds : np.ndarray
        Standard deviation vector of shape (latent_dims,)
    """
    if cell_idx >= len(meta_df):
        raise ValueError(f"Cell index {cell_idx} out of range (max: {len(meta_df) - 1})")

    lat_cols = _get_latent_columns(meta_df)
    std_cols = _get_std_columns(meta_df)

    if len(lat_cols) < latent_dims or len(std_cols) < latent_dims:
        raise ValueError(
            "Metadata does not contain enough latent mean/std columns to "
            f"match inferred dimensionality ({latent_dims})."
        )

    means = meta_df.iloc[cell_idx][lat_cols[:latent_dims]].to_numpy(dtype=float)
    stds = meta_df.iloc[cell_idx][std_cols[:latent_dims]].to_numpy(dtype=float)

    return means, stds


def compute_kl_divergence(means: np.ndarray, stds: np.ndarray) -> float:
    """
    Compute KL divergence between q(z|x)=N(mu, sigma^2) and p(z)=N(0, I).

    Mirrors the training-time calculation:
        KL = 0.5 * sum(mu^2 + sigma^2 - 1 - log(sigma^2))
    """
    if means.shape != stds.shape:
        raise ValueError("Means and standard deviations must share the same shape.")

    variances = np.square(stds)
    safe_variances = np.clip(variances, a_min=1e-12, a_max=None)
    kl = 0.5 * np.sum(np.square(means) + safe_variances - 1 - np.log(safe_variances))
    return float(kl)


def sample_latent_point(
    meta_df: pd.DataFrame, cell_idx: int, latent_dims: int
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a latent point from a real cell's latent distribution.
    
    Parameters
    ----------
    meta_df : pd.DataFrame
        DataFrame containing latent means and standard deviations
    cell_idx : int
        Index of the real cell to sample from
    latent_dims : int
        Number of latent dimensions
        
    Returns
    -------
    z_tensor : torch.Tensor
        Sampled latent vector of shape (1, latent_dims)
    means : np.ndarray
        Mean parameters used for sampling.
    stds : np.ndarray
        Standard deviation parameters used for sampling.
    latent_values : np.ndarray
        Sampled latent vector without batch dimension.
    """
    means, stds = _get_latent_stats(meta_df, cell_idx, latent_dims)

    # Sample a latent vector z ~ N(means, stds^2) for the given cell
    # Each dimension is independently sampled from its own normal dist
    z = np.random.normal(means, stds)
    
    # Convert numpy array to a torch tensor with an added batch dimension (shape: [1, latent_dims])
    z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
    
    # Print helpful summary statistics about the means and the sampled point
    print(f"Sampled latent point from cell {cell_idx}")
    print(f"  Mean range: [{means.min():.4f}, {means.max():.4f}]")
    print(f"  Sampled range: [{z.min():.4f}, {z.max():.4f}]")
    
    # Return the sampled tensor to be passed to the decoder
    return z_tensor, means, stds, z


def build_synthetic_metadata_entry(
    meta_df: pd.DataFrame,
    cell_idx: int,
    synthetic_id: str,
    latent_values: np.ndarray,
    stds: np.ndarray,
    kl_value: float,
    *,
    origin: str = "synthetic",
    source_cell_idx: Optional[int] = None,
    extra_metadata: Optional[dict] = None,
) -> tuple[str, dict]:
    """
    Prepare a metadata row for a synthetic cell.
    """
    base_row = meta_df.iloc[cell_idx].to_dict()
    entry = dict(base_row)

    lat_cols = _get_latent_columns(meta_df)
    std_cols = _get_std_columns(meta_df)

    for idx, col in enumerate(lat_cols):
        if idx >= latent_values.shape[0]:
            break
        entry[col] = latent_values[idx]
    for idx, col in enumerate(std_cols):
        if idx >= stds.shape[0]:
            break
        entry[col] = stds[idx]

    entry["kl_divergence"] = kl_value
    entry["source_cell_idx"] = (
        source_cell_idx if source_cell_idx is not None else cell_idx
    )
    entry["origin"] = origin
    entry["synthetic_id"] = synthetic_id
    if extra_metadata:
        entry.update(extra_metadata)

    return synthetic_id, entry


def plot_latent_umap_from_metadata(
    meta_df: pd.DataFrame, highlighted_ids: Sequence[str], output_dir: str
) -> Optional[str]:
    """
    Generate latent-space projections (UMAP + PCA) from latent metadata and save plots.
    Produces:
      - UMAP colored by origin
      - UMAP colored by cell type (if available)
      - PCA colored by origin
      - PCA colored by cell type (if available)
    """
    lat_cols = _get_latent_columns(meta_df)
    if not lat_cols:
        print("No latent columns found; skipping latent UMAP plot.")
        return None

    latent_matrix = meta_df[lat_cols].to_numpy()

    # UMAP embedding of the latent space
    if latent_matrix.shape[1] > 2:
        reducer = umap.UMAP(random_state=42)
        umap_embedding = reducer.fit_transform(latent_matrix)
    else:
        umap_embedding = latent_matrix

    # PCA embedding of the latent space (first two principal components)
    latent_centered = latent_matrix - latent_matrix.mean(axis=0, keepdims=True)
    if latent_centered.shape[1] > 1:
        # Use SVD to avoid depending on external PCA implementations
        _, _, vt = np.linalg.svd(latent_centered, full_matrices=False)
        pca_embedding = latent_centered @ vt[:2].T
    else:
        pca_embedding = latent_centered

    origin_series = (
        meta_df["origin"].fillna("real")
        if "origin" in meta_df.columns
        else pd.Series(["real"] * len(meta_df), index=meta_df.index)
    )

    # Use the `id` column for coloring if it exists
    id_series: Optional[pd.Series] = None
    if "id" in meta_df.columns:
        id_series = meta_df["id"]

    os.makedirs(output_dir, exist_ok=True)
    cmap = plt.get_cmap("tab10")

    def _plot_by_series(
        series: pd.Series,
        embedding: np.ndarray,
        title_prefix: str,
        filename: str,
        x_label: str,
        y_label: str,
    ) -> str:
        unique_vals = sorted(series.dropna().unique())
        fig, ax = plt.subplots(figsize=(8, 6))
        for idx, label in enumerate(unique_vals):
            mask = series == label
            color = cmap(idx % 10)
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=15,
                alpha=0.6,
                label=label,
                facecolor=color,
                edgecolor="none",
            )

        if highlighted_ids:
            highlight_mask = meta_df.index.isin(highlighted_ids)
            ax.scatter(
                embedding[highlight_mask, 0],
                embedding[highlight_mask, 1],
                s=40,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                label="synthetic_cells",
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title_prefix)
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        return plot_path

    # UMAP: primary latent-space plot colored by origin
    origin_umap_path = _plot_by_series(
        origin_series,
        umap_embedding,
        "Latent space UMAP by origin (real + synthetic)",
        "latent_umap_by_origin.png",
        x_label="UMAP-1" if umap_embedding.shape[1] > 1 else "Latent-1",
        y_label="UMAP-2" if umap_embedding.shape[1] > 1 else "Latent-2",
    )
    print(f"Saved latent UMAP (by origin) to: {origin_umap_path}")

    # UMAP: latent-space plot colored by `id`, if available
    if id_series is not None:
        id_umap_path = _plot_by_series(
            id_series,
            umap_embedding,
            "Latent space UMAP by id (real + synthetic)",
            "latent_umap_by_id.png",
            x_label="UMAP-1" if umap_embedding.shape[1] > 1 else "Latent-1",
            y_label="UMAP-2" if umap_embedding.shape[1] > 1 else "Latent-2",
        )
        print(f"Saved latent UMAP (by id) to: {id_umap_path}")

    # PCA: latent-space plot colored by origin
    origin_pca_path = _plot_by_series(
        origin_series,
        pca_embedding,
        "Latent space PCA by origin (real + synthetic)",
        "latent_pca_by_origin.png",
        x_label="PC1" if pca_embedding.shape[1] > 0 else "Latent-1",
        y_label="PC2" if pca_embedding.shape[1] > 1 else "Latent-2",
    )
    print(f"Saved latent PCA (by origin) to: {origin_pca_path}")

    # PCA: latent-space plot colored by `id`, if available
    if id_series is not None:
        id_pca_path = _plot_by_series(
            id_series,
            pca_embedding,
            "Latent space PCA by id (real + synthetic)",
            "latent_pca_by_id.png",
            x_label="PC1" if pca_embedding.shape[1] > 0 else "Latent-1",
            y_label="PC2" if pca_embedding.shape[1] > 1 else "Latent-2",
        )
        print(f"Saved latent PCA (by id) to: {id_pca_path}")

    # Maintain backwards-compatible return of the origin-colored UMAP path
    return origin_umap_path


def decode_latent(decoder: DecoderC, z: torch.Tensor) -> np.ndarray:
    """
    Decode a latent point into a gene-expression profile.
    
    Parameters
    ----------
    decoder : DecoderC
        The decoder model
    z : torch.Tensor
        Latent vector of shape (1, latent_dims)
        
    Returns
    -------
    decoded : np.ndarray
        Decoded gene-expression profile as 1D array
    """
    print("Decoding latent point...")
    
    with torch.no_grad():
        if hasattr(decoder, "pose") and decoder.pose:
            # Attempt to read pose_dims from attribute
            pose_dims = getattr(decoder, "pose_dims", None)
            
            # Infer pose_dims from the first linear layer if attribute is missing
            if pose_dims is None:
                for module in decoder.modules():
                    if isinstance(module, torch.nn.Linear):
                        pose_dims = module.in_features - z.size(1)
                        break
                        
            # Fallback if inference fails
            if pose_dims is None:
                pose_dims = 0
                
            if pose_dims > 0:
                x_pose = torch.zeros((z.size(0), pose_dims), dtype=z.dtype, device=z.device)
                decoded = decoder(z, x_pose)
            else:
                empty_pose = torch.empty((z.size(0), 0), dtype=z.dtype, device=z.device)
                decoded = decoder(z, empty_pose)
        else:
            try:
                decoded = decoder(z)
            except TypeError:
                decoded = decoder(z, None)
        
        decoded = decoded.cpu().numpy()
        
        if decoded.ndim > 1:
            decoded = decoded.squeeze()
        
        decoded = decoded.flatten()
    
    print(f"Decoded to expression profile of length {len(decoded)}")
    print(f"  Expression range: [{decoded.min():.4f}, {decoded.max():.4f}]")
    
    return decoded

def save_decoded_vector(decoded: np.ndarray, output_dir: str, cell_idx: int, synthetic_id: str):
    """
    Save decoded vector to disk.
    
    Parameters
    ----------
    decoded : np.ndarray
        Decoded gene-expression profile
    output_dir : str
        Output directory
    cell_idx : int
        Index of the source real cell
    synthetic_id : str
        Unique identifier for the synthetic cell
    """
    arrays_dir = os.path.join(output_dir, "output_arrays")
    os.makedirs(arrays_dir, exist_ok=True)
    
    filename = f"synthetic_decoded_vector_{synthetic_id}.npy"
    filepath = os.path.join(arrays_dir, filename)
    
    np.save(filepath, decoded)
    print(f"Saved decoded vector to: {filepath}")


def build_synthetic_cell(
    decoded: np.ndarray,
    synthetic_id: str,
    var_template: pd.DataFrame,
    source_celltype: str,
    cell_type_column_name: str,
    source_cell_idx: Optional[int] = None,
    *,
    origin: str = "synthetic",
    extra_obs: Optional[dict] = None,
) -> ad.AnnData:
    """
    Create a single-cell AnnData object for a decoded synthetic profile.
    
    Parameters
    ----------
    decoded : np.ndarray
        Decoded gene-expression profile
    synthetic_id : str
        Unique identifier for the synthetic cell
    var_template : pd.DataFrame
        Gene metadata copied from the reference AnnData
    source_celltype : str
        Normalized cell type inherited from the source real cell
    cell_type_column_name : str
        Observation column used for VAE class labels
        
    Returns
    -------
    ad.AnnData
        One-row AnnData containing the synthetic expression vector
    """
    decoded_expanded = decoded.reshape(1, -1)
    
    obs_payload = {
        "origin": [origin],
        "is_synthetic": [True],
        cell_type_column_name: [source_celltype],
    }
    if source_cell_idx is not None:
        obs_payload["source_cell_idx"] = [source_cell_idx]
    if extra_obs:
        for key, value in extra_obs.items():
            obs_payload[key] = [value]
    
    synthetic_obs = pd.DataFrame(obs_payload, index=[synthetic_id])
    
    synthetic_adata = ad.AnnData(
        X=decoded_expanded,
        obs=synthetic_obs,
        var=var_template.copy(),
    )
    
    return synthetic_adata


def recompute_umap(
    adata: ad.AnnData,
    output_dir: str,
    save_prefix: str = "synthetic",
    primary_color_key: str = "celltype_level_1",
):
    """
    Recompute PCA, neighbors, and UMAP, then save plot.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with synthetic cells
    output_dir : str
        Output directory for saving plot
    save_prefix : str
        Prefix for saved plot filename
    primary_color_key : str
        Observation column to use for the primary UMAP coloring.
    """
    print("Recomputing PCA, neighbors, and UMAP...")
    
    # Compute PCA
    print("  Computing PCA...")
    sc.tl.pca(adata, svd_solver="arpack")
    
    # Compute neighbors
    print("  Computing neighbors...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    
    # Compute UMAP
    print("  Computing UMAP...")
    sc.tl.umap(adata)
    
    # Save UMAP plots
    plot_dir = os.path.join(output_dir, "umap_plots")
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d%m%y_%H%M")

    # Always plot both the requested key and origin
    requested_keys = [primary_color_key, "origin"]
    seen_keys: set[str] = set()
    ordered_keys: list[str] = []
    for key in requested_keys:
        if key not in seen_keys:
            ordered_keys.append(key)
            seen_keys.add(key)

    import matplotlib.pyplot as plt

    # UMAP projections
    for color_key in ordered_keys:
        if color_key not in adata.obs.columns:
            print(
                f"  Skipping UMAP colored by '{color_key}' "
                "(column not found in adata.obs)."
            )
            continue

        print(f"  Saving UMAP plot colored by '{color_key}'.")
        sc.pl.umap(adata, color=color_key, show=False)
        safe_key = color_key.replace("/", "_")
        plot_path = os.path.join(
            plot_dir, f"{save_prefix}_umap_{safe_key}_{timestamp}.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    # PCA projections
    for color_key in ordered_keys:
        if color_key not in adata.obs.columns:
            print(
                f"  Skipping PCA colored by '{color_key}' "
                "(column not found in adata.obs)."
            )
            continue

        print(f"  Saving PCA plot colored by '{color_key}'.")
        sc.pl.pca(adata, color=color_key, show=False)
        safe_key = color_key.replace("/", "_")
        pca_path = os.path.join(
            plot_dir, f"{save_prefix}_pca_{safe_key}_{timestamp}.png"
        )
        plt.savefig(pca_path, dpi=300, bbox_inches="tight")
        plt.close()

    print("UMAP computation complete.")



# ============================================================================
# Interpolation helpers
# ============================================================================


def interpolate_latents(
    start: np.ndarray, end: np.ndarray, n_interp: int
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate spherical linear interpolation (slerp) points between two vectors.
    Returns the interpolated latent vectors and their normalized positions t.
    """
    if n_interp < 2:
        raise ValueError("n_interp must be at least 2 to perform interpolation.")

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != end.shape:
        raise ValueError("Latent vectors must have matching shapes for interpolation.")

    t_values = np.linspace(0.0, 1.0, n_interp)
    start_norm = np.linalg.norm(start)
    end_norm = np.linalg.norm(end)

    if start_norm == 0 or end_norm == 0:
        # Fall back to linear interpolation if either vector is zero
        latent_points = np.outer(1 - t_values, start) + np.outer(t_values, end)
        return latent_points, t_values

    start_unit = start / start_norm
    end_unit = end / end_norm
    dot = np.clip(np.dot(start_unit, end_unit), -1.0, 1.0)
    theta = np.arccos(dot)

    if np.isclose(theta, 0):
        latent_points = np.outer(1 - t_values, start) + np.outer(t_values, end)
        return latent_points, t_values

    sin_theta = np.sin(theta)
    latent_points = []
    for t in t_values:
        factor0 = np.sin((1 - t) * theta) / sin_theta
        factor1 = np.sin(t * theta) / sin_theta
        point = factor0 * start + factor1 * end
        latent_points.append(point)

    return np.vstack(latent_points), t_values

def save_anndata(adata: ad.AnnData, output_path: str):
    """
    Save AnnData object to disk.
    
    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to save
    output_path : str
        Path to save the .h5ad file
    """
    print(f"Saving AnnData to: {output_path}")
    
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    adata.write(output_path)
    
    print("AnnData saved successfully.")


# ============================================================================
# Main execution
# ============================================================================


def main(
    adata_path: str,
    checkpoint_path: str,
    meta_path: str,
    cell_indices: Optional[Sequence[int]] = None,
    output_suffix: Optional[str] = None,
    output_root: Optional[str] = None,
    output_h5ad: Optional[str] = None,
    skip_h5ad: bool = False,
    save_npy: bool = False,
    generate_umap: bool = True,
    cell_type_column_name: str = "celltype_level_1",
    umap_color_key: Optional[str] = None,
    *,
    interpolate: bool = False,
    n_interp: int = 5,
):
    """
    Main function to generate synthetic cells.
    
    Parameters
    ----------
    adata_path : str
        Path to the AnnData (.h5ad) file
    checkpoint_path : str
        Path to the decoder checkpoint (.pt)
    meta_path : str
        Path to the latent metadata (.pkl)
    cell_indices : Sequence[int], optional
        Meta-pickle row indices of real cells to sample from (default: [0])
    output_suffix : Optional[str]
        Suffix for output directory (default: timestamp)
    output_root : Optional[str]
        Base directory for outputs (default: derived from checkpoint path)
    output_h5ad : Optional[str]
        Path for combined h5ad output (default: output_dir/adata_with_synthetic.h5ad)
    skip_h5ad : bool
        Skip writing the combined h5ad (default: False)
    save_npy : bool
        Also save per-cell decoded vectors as .npy files (default: False)
    generate_umap : bool
        Whether to recompute and save UMAP plots (default: True)
    cell_type_column_name : str
        Observation column for cell-type labels (must match VAE training config)
    umap_color_key : Optional[str]
        adata.obs column to color the primary UMAP plot (defaults to cell_type_column_name)
    """
    if umap_color_key is None:
        umap_color_key = cell_type_column_name
    print("=" * 70)
    print("Synthetic Cell Generation Script")
    print("=" * 70)
    
    if interpolate:
        if not cell_indices or len(cell_indices) != 2:
            raise ValueError(
                "Interpolation mode requires exactly two --cell_idx entries."
            )
        if n_interp < 2:
            raise ValueError("Interpolation mode requires n_interp >= 2.")
        print("Interpolation mode enabled.")
        print(
            f"Preparing to generate {n_interp} interpolated cells "
            f"between indices {cell_indices[0]} and {cell_indices[1]}."
        )
    else:
        if not cell_indices:
            cell_indices = [0]
        cell_indices = list(cell_indices)
        print(f"Preparing to generate {len(cell_indices)} synthetic cells.")
    
    # Set output directory root near the checkpoint if not provided
    if output_root is None:
        output_root = os.path.dirname(os.path.dirname(checkpoint_path))
    
    # Determine output suffix
    if output_suffix is None:
        output_suffix = datetime.now().strftime("%d%m%y_%H%M")
    
    run_dir_name = f"generative_out_{output_suffix}"
    output_dir = os.path.join(output_root, run_dir_name)
    
    # Set device
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load decoder
    decoder = load_decoder(checkpoint_path, device)
    
    # Load latent metadata
    meta_df = load_latent_metadata(meta_path)
    
    # Infer latent dimensions
    latent_dims = infer_latent_dims(meta_df)
    
    # Load AnnData
    print(f"Loading AnnData from: {adata_path}")
    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")
    
    adata = ad.read_h5ad(adata_path)
    print(f"Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes")

    if cell_type_column_name not in adata.obs.columns:
        raise ValueError(
            f"Column '{cell_type_column_name}' not found in input adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    n_vars = adata.n_vars

    # Ensure origin metadata exists before appending new cells
    if "origin" not in adata.obs.columns:
        adata.obs["origin"] = "real"
    if "is_synthetic" not in adata.obs.columns:
        adata.obs["is_synthetic"] = False
    
    var_template = adata.var
    synthetic_cells: list[ad.AnnData] = []
    
    synthetic_metadata_entries: list[tuple[str, dict]] = []
    synthetic_ids: list[str] = []

    if interpolate:
        cell_a, cell_b = cell_indices
        means_a, stds_a = _get_latent_stats(meta_df, cell_a, latent_dims)
        means_b, stds_b = _get_latent_stats(meta_df, cell_b, latent_dims)

        latent_points, t_positions = interpolate_latents(means_a, means_b, n_interp)
        std_points = np.array(
            [(1 - t) * stds_a + t * stds_b for t in t_positions], dtype=float
        )

        for i, (latent_vec, std_vec, t_val) in enumerate(
            zip(latent_points, std_points, t_positions)
        ):
            print("-" * 40)
            print(
                f"Interpolating between cells {cell_a} and {cell_b} "
                f"(t={t_val:.2f})"
            )
            z_tensor = torch.tensor(latent_vec, dtype=torch.float32).unsqueeze(0)
            decoded = decode_latent(decoder, z_tensor)

            synthetic_id = (
                f"interp_{i:05d}_src{cell_a:06d}_to_{cell_b:06d}"
            )
            synthetic_ids.append(synthetic_id)
            kl_value = compute_kl_divergence(latent_vec, std_vec)
            base_idx = cell_a if t_val <= 0.5 else cell_b
            extra_meta = {
                "interpolation_pair": (cell_a, cell_b),
                "t_position": float(t_val),
            }
            meta_entry = build_synthetic_metadata_entry(
                meta_df=meta_df,
                cell_idx=base_idx,
                synthetic_id=synthetic_id,
                latent_values=latent_vec,
                stds=std_vec,
                kl_value=kl_value,
                origin="interpolation",
                source_cell_idx=None,
                extra_metadata=extra_meta,
            )
            synthetic_metadata_entries.append(meta_entry)

            source_celltype = get_source_celltype(
                adata, meta_df, base_idx, cell_type_column_name
            )
            if save_npy:
                save_decoded_vector(decoded, output_dir, base_idx, synthetic_id)
            synthetic_cells.append(
                build_synthetic_cell(
                    decoded=decoded,
                    synthetic_id=synthetic_id,
                    var_template=var_template,
                    source_celltype=source_celltype,
                    cell_type_column_name=cell_type_column_name,
                    source_cell_idx=None,
                    origin="interpolation",
                    extra_obs={
                        "interpolation_pair": f"{cell_a}-{cell_b}",
                        "t_position": t_val,
                    },
                )
            )
    else:
        for i, cell_idx in enumerate(cell_indices):
            print("-" * 40)
            print(f"Processing source cell index: {cell_idx}")
            
            # Sample latent point
            z, means, stds, z_np = sample_latent_point(meta_df, cell_idx, latent_dims)
            
            # Decode latent point
            decoded = decode_latent(decoder, z)
            
            # Generate synthetic ID (includes both loop count and source idx)
            synthetic_id = f"synthetic_{i:05d}_src{cell_idx:06d}"
            synthetic_ids.append(synthetic_id)
            kl_value = compute_kl_divergence(means, stds)
            meta_entry = build_synthetic_metadata_entry(
                meta_df=meta_df,
                cell_idx=cell_idx,
                synthetic_id=synthetic_id,
                latent_values=z_np,
                stds=stds,
                kl_value=kl_value,
            )
            synthetic_metadata_entries.append(meta_entry)

            source_celltype = get_source_celltype(
                adata, meta_df, cell_idx, cell_type_column_name
            )
            if save_npy:
                save_decoded_vector(decoded, output_dir, cell_idx, synthetic_id)

            # Build synthetic AnnData (defer concatenation for efficiency)
            synthetic_cells.append(
                build_synthetic_cell(
                    decoded=decoded,
                    synthetic_id=synthetic_id,
                    var_template=var_template,
                    source_celltype=source_celltype,
                    cell_type_column_name=cell_type_column_name,
                    source_cell_idx=cell_idx,
                )
            )
    updated_meta_path: Optional[str] = None
    if synthetic_metadata_entries:
        new_indices = [idx for idx, _ in synthetic_metadata_entries]
        new_rows = [entry for _, entry in synthetic_metadata_entries]
        synthetic_meta_df = pd.DataFrame(new_rows, index=new_indices)
        meta_df = pd.concat([meta_df, synthetic_meta_df], axis=0)
        metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        meta_filename = f"{Path(meta_path).stem}_with_synthetic.pkl"
        updated_meta_path = os.path.join(metadata_dir, meta_filename)
        meta_df.to_pickle(updated_meta_path)
        print(f"Saved updated latent metadata (with KL) to: {updated_meta_path}")

    if synthetic_cells:
        print("Concatenating synthetic cells into AnnData...")
        new_cells = ad.concat(synthetic_cells, axis=0, join="outer")
        adata = ad.concat([adata, new_cells], axis=0, join="outer")
        adata = finalize_combined_adata(adata, cell_type_column_name)

        print(
            f"Added {len(synthetic_cells)} synthetic cells. "
            f"Total cells now: {adata.n_obs}"
        )
    else:
        print("No synthetic cells were generated (empty cell_indices).")
        adata = finalize_combined_adata(adata, cell_type_column_name)
    latent_umap_path = None
    if synthetic_metadata_entries:
        latent_umap_path = plot_latent_umap_from_metadata(
            meta_df,
            highlighted_ids=synthetic_ids,
            output_dir=os.path.join(output_dir, "latent_umap"),
        )
    
    # Recompute UMAP if requested
    if generate_umap:
        recompute_umap(
            adata,
            output_dir,
            save_prefix="synthetic",
            primary_color_key=umap_color_key,
        )
    
    # Save combined AnnData (primary output)
    if output_h5ad is None:
        output_adata_path = os.path.join(output_dir, "adata_with_synthetic.h5ad")
    else:
        output_adata_path = output_h5ad

    if not skip_h5ad:
        save_anndata(adata, output_adata_path)
        validate_h5ad_output(
            output_adata_path,
            cell_type_column_name,
            expected_n_obs=adata.n_obs,
            expected_n_vars=n_vars,
        )
    
    print("=" * 70)
    print("Synthetic cell generation complete!")
    print("=" * 70)
    print("\nOutput summary:")
    if not skip_h5ad:
        print(f"  - Combined AnnData: {output_adata_path}")
    else:
        print("  - Combined AnnData: skipped (--skip_h5ad)")
    if save_npy:
        print(
            "  - Decoded vectors: "
            f"{output_dir}/output_arrays/synthetic_decoded_vector_*.npy"
        )
    if generate_umap:
        print(f"  - UMAP plots: {output_dir}/umap_plots/*.png")
    else:
        print("  - UMAP plot: skipped")
    if updated_meta_path:
        print(f"  - Latent metadata with synthetics: {updated_meta_path}")
    else:
        print("  - Latent metadata: unchanged")
    if latent_umap_path:
        print(f"  - Latent space UMAP: {latent_umap_path}")
    else:
        print("  - Latent space UMAP: skipped (no synthetic metadata)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic scRNA-seq cells using VAE decoder"
    )
    parser.add_argument(
        "--cell_idx",
        type=int,
        nargs="+",
        metavar="IDX",
        default=[0],
        help="One or more indices of real cells to sample from (default: 0)"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="Suffix for output directory (default: current time_date stamp)"
    )
    parser.add_argument(
        "--adata_path",
        type=str,
        required=True,
        help="Path to AnnData (.h5ad) file"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to decoder checkpoint (.pt)"
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        required=True,
        help="Path to latent metadata (.pkl)"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional base directory for outputs (default: checkpoint parent)"
    )
    parser.add_argument(
        "--output_h5ad",
        type=str,
        default=None,
        help="Path for combined h5ad output (default: output_dir/adata_with_synthetic.h5ad)",
    )
    parser.add_argument(
        "--skip_h5ad",
        action="store_true",
        help="Skip writing the combined h5ad file",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Also save per-cell decoded vectors as .npy files (default: False)",
    )
    parser.add_argument(
        "--skip_umap",
        action="store_true",
        help="Skip recomputing UMAP/plot (UMAP enabled by default)",
    )
    parser.add_argument(
        "--cell_type_column_name",
        "-ctcn",
        type=str,
        default="celltype_level_1",
        help=(
            "Observation column for cell-type labels "
            "(must match VAE training config; default: celltype_level_1)"
        ),
    )
    parser.add_argument(
        "--umap_color_key",
        type=str,
        default=None,
        help=(
            "Observation column to color the primary UMAP plot "
            "(defaults to --cell_type_column_name). "
            "An additional plot colored by 'origin' is always produced."
        ),
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Enable latent interpolation mode (requires exactly two --cell_idx entries)",
    )
    parser.add_argument(
        "--n_interp",
        type=int,
        default=5,
        help="Number of interpolation points (used only when --interpolate is set)",
    )
    
    args = parser.parse_args()
    
    main(
        adata_path=args.adata_path,
        checkpoint_path=args.checkpoint_path,
        meta_path=args.meta_path,
        cell_indices=args.cell_idx,
        output_suffix=args.output_suffix,
        output_root=args.output_root,
        output_h5ad=args.output_h5ad,
        skip_h5ad=args.skip_h5ad,
        save_npy=args.save_npy,
        generate_umap=not args.skip_umap,
        cell_type_column_name=args.cell_type_column_name,
        umap_color_key=args.umap_color_key,
        interpolate=args.interpolate,
        n_interp=args.n_interp,
    )

