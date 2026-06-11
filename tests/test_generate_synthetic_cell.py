import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from avae.data import Dataset_h5ad_reader

MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "omics" / "generate_synthetic_cell.py"
)
_spec = importlib.util.spec_from_file_location("generate_synthetic_cell", MODULE_PATH)
gsc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(gsc)


def _make_reference_adata(n_obs: int = 3, n_vars: int = 5) -> ad.AnnData:
    X = np.arange(n_obs * n_vars, dtype=float).reshape(n_obs, n_vars)
    obs = pd.DataFrame(
        {
            "celltype_level_1": ["B cell", "T cell", "B cell, memory"],
            "origin": ["real"] * n_obs,
            "is_synthetic": [False] * n_obs,
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_vars)])
    return ad.AnnData(X=X, obs=obs, var=var)


def _make_meta_df(n_rows: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        row = {"filename": str(i), "id": f"type_{i}", "mode": "trn"}
        for d in range(2):
            row[f"lat{d}"] = float(i + d)
            row[f"std-{d}"] = 0.1
        rows.append(row)
    return pd.DataFrame(rows)


class GenerateSyntheticCellHelpersTest(unittest.TestCase):
    def test_normalize_celltype_label(self):
        self.assertEqual(gsc.normalize_celltype_label("B cell"), "B-cell")
        self.assertEqual(
            gsc.normalize_celltype_label("B cell, memory"), "B-cell--memory"
        )

    def test_resolve_h5ad_row(self):
        meta_df = _make_meta_df()
        self.assertEqual(gsc.resolve_h5ad_row(meta_df, 1, n_obs=3), 1)

        with self.assertRaises(ValueError):
            gsc.resolve_h5ad_row(meta_df, 0, n_obs=1)

    def test_get_source_celltype(self):
        adata = _make_reference_adata()
        meta_df = _make_meta_df()
        self.assertEqual(
            gsc.get_source_celltype(adata, meta_df, 0, "celltype_level_1"),
            "B-cell",
        )
        self.assertEqual(
            gsc.get_source_celltype(adata, meta_df, 2, "celltype_level_1"),
            "B-cell--memory",
        )

    def test_build_synthetic_cell_inherits_type(self):
        adata = _make_reference_adata()
        decoded = np.ones(adata.n_vars)
        synthetic = gsc.build_synthetic_cell(
            decoded=decoded,
            synthetic_id="syn_000",
            var_template=adata.var,
            source_celltype="T-cell",
            cell_type_column_name="celltype_level_1",
            source_cell_idx=1,
        )
        self.assertEqual(synthetic.obs["celltype_level_1"].iloc[0], "T-cell")
        self.assertTrue(synthetic.obs["is_synthetic"].iloc[0])
        self.assertEqual(synthetic.obs["origin"].iloc[0], "synthetic")

    def test_finalize_combined_adata(self):
        real = _make_reference_adata()
        synthetic = gsc.build_synthetic_cell(
            decoded=np.zeros(real.n_vars),
            synthetic_id="syn_000",
            var_template=real.var,
            source_celltype="B-cell",
            cell_type_column_name="celltype_level_1",
        )
        combined = ad.concat([real, synthetic], axis=0, join="outer")
        combined = gsc.finalize_combined_adata(combined, "celltype_level_1")

        self.assertIn("origin", combined.obs.columns)
        self.assertIn("is_synthetic", combined.obs.columns)
        self.assertFalse(combined.obs["is_synthetic"].iloc[0])
        self.assertTrue(combined.obs["is_synthetic"].iloc[-1])


class GenerateSyntheticCellH5adRoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="gsc_h5ad_")

    def tearDown(self) -> None:
        if os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir)

    def test_h5ad_round_trip_with_dataset_reader(self):
        real = _make_reference_adata()
        synthetic = gsc.build_synthetic_cell(
            decoded=np.full(real.n_vars, 2.0),
            synthetic_id="syn_000",
            var_template=real.var,
            source_celltype="T-cell",
            cell_type_column_name="celltype_level_1",
            source_cell_idx=1,
        )
        combined = ad.concat([real, synthetic], axis=0, join="outer")
        combined = gsc.finalize_combined_adata(combined, "celltype_level_1")

        output_path = os.path.join(self._tmpdir, "combined.h5ad")
        combined.write(output_path)
        gsc.validate_h5ad_output(
            output_path,
            "celltype_level_1",
            expected_n_obs=combined.n_obs,
            expected_n_vars=combined.n_vars,
        )

        dataset = Dataset_h5ad_reader(
            datafile=output_path,
            cell_type_column_name="celltype_level_1",
        )
        self.assertEqual(len(dataset), combined.n_obs)
        self.assertEqual(dataset.dim(), combined.n_vars)

        x, label, _, meta = dataset[combined.n_obs - 1]
        self.assertEqual(label, "T-cell")
        self.assertEqual(meta["id"], "T-cell")
