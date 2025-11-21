#!/usr/bin/env python3
"""
Script to generate synthetic scRNA-seq gene-expression profiles using a trained VAE decoder.

This script:
1. Loads a trained DecoderC from a checkpoint
2. Samples latent points from real cell latent distributions
3. Decodes latent points into synthetic gene-expression profiles
4. Inserts synthetic cells into an existing AnnData object
5. Recomputes UMAP embeddings with the synthetic cells
6. Saves outputs (decoded vectors, UMAP plot, modified AnnData)
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
    entry["source_cell_idx"] = cell_idx
    entry["origin"] = "synthetic"
    entry["synthetic_id"] = synthetic_id

    return synthetic_id, entry


def plot_latent_umap_from_metadata(
    meta_df: pd.DataFrame, highlighted_ids: Sequence[str], output_dir: str
) -> Optional[str]:
    """
    Generate a UMAP projection from latent metadata and save plot.
    """
    lat_cols = _get_latent_columns(meta_df)
    if not lat_cols:
        print("No latent columns found; skipping latent UMAP plot.")
        return None

    latent_matrix = meta_df[lat_cols].to_numpy()
    if latent_matrix.shape[1] > 2:
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(latent_matrix)
    else:
        embedding = latent_matrix

    origin_series = (
        meta_df["origin"].fillna("real") if "origin" in meta_df.columns else pd.Series(
            ["real"] * len(meta_df), index=meta_df.index
        )
    )
    unique_origins = sorted(origin_series.unique())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(unique_origins):
        mask = origin_series == label
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

    ax.set_xlabel("UMAP-1" if embedding.shape[1] > 1 else "Latent-1")
    ax.set_ylabel("UMAP-2" if embedding.shape[1] > 1 else "Latent-2")
    ax.set_title("Latent space UMAP (real + synthetic)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "latent_umap.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"Saved latent UMAP plot to: {plot_path}")
    return plot_path


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
        # DecoderC expects (latent, pose) but pose can be None if pose_dims=0
        # Check if decoder has pose attribute
        if hasattr(decoder, "pose") and decoder.pose:
            # This decoder uses pose, but we'll pass None for now
            # If needed, we can sample pose separately
            decoded = decoder(z, None)
        else:
            decoded = decoder(z, None)
        
        # DecoderC outputs shape (N, 1, input_size), so we need to squeeze
        decoded = decoded.cpu().numpy()
        
        # Remove batch and channel dimensions if present
        if decoded.ndim > 1:
            decoded = decoded.squeeze()
        
        # Ensure 1D output
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
    source_cell_idx: Optional[int] = None,
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
        
    Returns
    -------
    ad.AnnData
        One-row AnnData containing the synthetic expression vector
    """
    decoded_expanded = decoded.reshape(1, -1)
    
    obs_payload = {"origin": ["aVAE"]}
    if source_cell_idx is not None:
        obs_payload["source_cell_idx"] = [source_cell_idx]
    
    synthetic_obs = pd.DataFrame(obs_payload, index=[synthetic_id])
    
    synthetic_adata = ad.AnnData(
        X=decoded_expanded,
        obs=synthetic_obs,
        var=var_template.copy(),
    )
    
    return synthetic_adata


def recompute_umap(adata: ad.AnnData, output_dir: str, save_prefix: str = "synthetic"):
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
    
    # Save UMAP plot
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d%m%y_%H%M")
    plot_path = os.path.join(output_dir, f"{save_prefix}_umap_{timestamp}.png")

    
    print(f"  Saving UMAP plot to: {plot_path}")
    
    # Use matplotlib to save directly to desired location
    import matplotlib.pyplot as plt
    
    # Plot UMAP
    color_key = "celltype_level_1" if "celltype_level_1" in adata.obs.columns else "origin"
    sc.pl.umap(adata, color=color_key, show=False)
    
    # Save the current figure
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print("UMAP computation complete.")


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
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
    save_h5ad: bool = False,
    generate_umap: bool = True,
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
        One or more indices of real cells to sample from (default: [0])
    output_suffix : Optional[str]
        Suffix for output directory (default: timestamp)
    output_root : Optional[str]
        Base directory for outputs (default: derived from checkpoint path)
    save_h5ad : bool
        Whether to persist the augmented AnnData object (default: False)
    generate_umap : bool
        Whether to recompute and save UMAP plots (default: True)
    """
    print("=" * 70)
    print("Synthetic Cell Generation Script")
    print("=" * 70)
    
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
    
    # Ensure origin metadata exists before appending new cells
    if "origin" not in adata.obs.columns:
        adata.obs["origin"] = "real"
    
    var_template = adata.var
    synthetic_cells: list[ad.AnnData] = []
    
    synthetic_metadata_entries: list[tuple[str, dict]] = []
    synthetic_ids: list[str] = []

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
        
        # Save decoded vector
        save_decoded_vector(decoded, output_dir, cell_idx, synthetic_id)
        
        # Build synthetic AnnData (defer concatenation for efficiency)
        synthetic_cells.append(
            build_synthetic_cell(
                decoded=decoded,
                synthetic_id=synthetic_id,
                var_template=var_template,
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
        new_obs_names = new_cells.obs_names
        adata = ad.concat([adata, new_cells], axis=0, join="outer")
        
        # Ensure categorical metadata columns include the synthetic label
        for meta_col in ("cell_type", "celltype_level_1"):
            if meta_col in adata.obs.columns:
                series = adata.obs[meta_col]
                if pd.api.types.is_categorical_dtype(series):
                    adata.obs[meta_col] = series.cat.add_categories(["aVAE"])
                adata.obs.loc[new_obs_names, meta_col] = "aVAE"
        
        print(
            f"Added {len(synthetic_cells)} synthetic cells. "
            f"Total cells now: {adata.n_obs}"
        )
    else:
        print("No synthetic cells were generated (empty cell_indices).")
    latent_umap_path = None
    if synthetic_metadata_entries:
        latent_umap_path = plot_latent_umap_from_metadata(
            meta_df,
            highlighted_ids=synthetic_ids,
            output_dir=os.path.join(output_dir, "latent_umap"),
        )
    
    # Recompute UMAP if requested
    if generate_umap:
        recompute_umap(adata, output_dir, save_prefix="synthetic")
    
    # Save modified AnnData if requested
    output_adata_path = os.path.join(output_dir, "adata_with_synthetic.h5ad")
    if save_h5ad:
        save_anndata(adata, output_adata_path)
    
    print("=" * 70)
    print("Synthetic cell generation complete!")
    print("=" * 70)
    print("\nOutput summary:")
    print("  - Decoded vectors: "
          f"{output_dir}/output_arrays/synthetic_decoded_vector_*.npy")
    if generate_umap:
        print(f"  - UMAP plot: {output_dir}/synthetic_umap_*")
    else:
        print("  - UMAP plot: skipped")
    if save_h5ad:
        print(f"  - Modified AnnData: {output_adata_path}")
    else:
        print("  - Modified AnnData: skipped (use --save_h5ad to enable)")
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
        "--save_h5ad",
        action="store_true",
        help="Persist the AnnData with synthetic cells (default: False)"
    )
    parser.add_argument(
        "--skip_umap",
        action="store_true",
        help="Skip recomputing UMAP/plot (UMAP enabled by default)"
    )
    
    args = parser.parse_args()
    
    main(
        adata_path=args.adata_path,
        checkpoint_path=args.checkpoint_path,
        meta_path=args.meta_path,
        cell_indices=args.cell_idx,
        output_suffix=args.output_suffix,
        output_root=args.output_root,
        save_h5ad=args.save_h5ad,
        generate_umap=not args.skip_umap,
    )

