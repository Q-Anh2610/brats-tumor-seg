"""Synthetic tests only: these are not results on the 55 real patients."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion, generate_binary_structure
from src.eval.metrics_3d import metrics_3d
from src.eval.nifti_io import checked_array, grid_mm, same_grid, find_raw_file
from src.eval.evaluate import summary_rows


def create_fixture(root):
    root = Path(root)
    raw, pred = root / "raw", root / "pred"
    raw.mkdir(parents=True, exist_ok=True)
    pred.mkdir(parents=True, exist_ok=True)
    # Original LPS orientation, unequal spacing catches axis/spacing mistakes.
    affine = np.diag([-1., -1.5, 2., 1.])
    affine[:3, 3] = [18., 28., 0.]
    ids = []
    for i in range(6):
        cid = f"BraTS20_Training_{i + 1:03d}"
        ids.append(cid)
        case = raw / cid
        case.mkdir(exist_ok=True)
        gt = np.zeros((24, 28, 16), dtype=np.uint8)
        gt[7:16, 8:19, 4:12] = 2
        pp = np.roll((gt > 0).astype(np.uint8), i, axis=0)
        xx, yy, zz = np.indices(gt.shape)
        mri = ((xx + 2 * yy + zz + 20) * (((xx-12)**2/100+(yy-14)**2/144+(zz-8)**2/49) < 1)).astype(np.float32)
        for path, array in [(case / f"{cid}_seg.nii.gz", gt),
                            (case / f"{cid}_flair.nii.gz", mri),
                            (pred / f"{cid}_pred_wt.nii.gz", pp)]:
            img = nib.Nifti1Image(array, affine)
            img.header.set_xyzt_units("mm")
            nib.save(img, path)
    split = root / "splits.json"
    split.write_text(json.dumps({"counts": {"test": len(ids)}, "splits": {"test": ids}}), encoding="utf-8")
    return raw, pred, split, ids


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((9, 10, 11), dtype=bool)
        self.mask[2:6, 3:7, 1:6] = True

    def test_identical(self):
        r = metrics_3d(self.mask, self.mask, [1, 2, 3])
        self.assertEqual((r["dice"], r["iou"], r["hd95_mm"]), (1., 1., 0.))

    def test_both_empty(self):
        z = np.zeros_like(self.mask)
        r = metrics_3d(z, z, [1, 1, 1])
        self.assertEqual((r["dice"], r["iou"], r["hd95_mm"]), (1., 1., 0.))

    def test_one_empty_each_direction(self):
        for p, g in [(self.mask, np.zeros_like(self.mask)), (np.zeros_like(self.mask), self.mask)]:
            r = metrics_3d(p, g, [1, 1, 1])
            self.assertEqual(r["dice"], 0)
            self.assertEqual(r["iou"], 0)
            self.assertTrue(np.isinf(r["hd95_mm"]))

    def test_anisotropic_single_voxel_translation(self):
        p, g = np.zeros((7, 8, 9)), np.zeros((7, 8, 9))
        p[2, 2, 2], g[2, 2, 4] = 1, 1
        self.assertEqual(metrics_3d(p, g, [1, 2, 3])["hd95_mm"], 6.)

    def test_hd95_against_brute_force_boundary_distances(self):
        p = self.mask.copy()
        p[0, 0, 0] = True
        g = np.roll(self.mask, 1, axis=1)
        structure = generate_binary_structure(3, 1)
        bp = p & ~binary_erosion(p, structure=structure, border_value=0)
        bg = g & ~binary_erosion(g, structure=structure, border_value=0)
        dist = cdist(np.argwhere(bp) * [1, 2, 3], np.argwhere(bg) * [1, 2, 3])
        expected = max(np.percentile(dist.min(axis=0), 95), np.percentile(dist.min(axis=1), 95))
        self.assertAlmostEqual(metrics_3d(p, g, [1, 2, 3])["hd95_mm"], expected)

    def test_overlap(self):
        g, p = np.zeros((4, 4, 4)), np.zeros((4, 4, 4))
        g[0:2, 0, 0] = 1
        p[1:3, 0, 0] = 1
        r = metrics_3d(p, g, [1, 1, 1])
        self.assertEqual(r["dice"], .5)
        self.assertAlmostEqual(r["iou"], 1/3)

    def test_nonbinary_rejected(self):
        with self.assertRaises(ValueError):
            metrics_3d(self.mask.astype(float) * .5, self.mask, [1, 1, 1])

    def test_summary_keeps_infinity_and_sample_sd(self):
        rows = [{"dice": .5, "iou": 1/3, "hd95_mm": 2.},
                {"dice": 1., "iou": 1., "hd95_mm": float("inf")}]
        summary = summary_rows(rows)
        self.assertAlmostEqual(summary[0]["sd_all"], np.std([.5, 1.], ddof=1))
        self.assertEqual(summary[2]["mean_all"], "inf")
        self.assertEqual(summary[2]["n_infinite"], 1)
        self.assertEqual(summary[2]["mean_finite"], 2.)
        self.assertEqual(summary[2]["sd_finite"], "N/A")
        self.assertEqual(summary_rows(rows[:1])[0]["sd_all"], "N/A")


class GridTests(unittest.TestCase):
    def image(self, unit="mm", affine=None):
        img = nib.Nifti1Image(np.zeros((5, 6, 7)), np.eye(4) if affine is None else affine)
        img.header.set_xyzt_units(unit)
        return img

    def test_unknown_requires_explicit_assumption(self):
        img = self.image("unknown")
        with self.assertRaises(ValueError):
            grid_mm(img)
        self.assertEqual(grid_mm(img, True)[2], "unknown")

    def test_units_convert_to_same_physical_grid(self):
        mm = self.image()
        aff = np.diag([.001, .001, .001, 1.])
        meter = self.image("meter", aff)
        np.testing.assert_allclose(same_grid(mm, meter), [1, 1, 1], atol=1e-5)

    def test_affine_translation_rejected(self):
        aff = np.eye(4)
        aff[0, 3] = 3.
        with self.assertRaises(ValueError):
            same_grid(self.image(), self.image(affine=aff))

    def test_shear_rejected(self):
        aff = np.eye(4)
        aff[0, 1] = .2
        with self.assertRaises(ValueError):
            grid_mm(self.image(affine=aff))

    def test_fractional_label_and_nan_rejected(self):
        for value in (.5, float("nan")):
            img = nib.Nifti1Image(np.full((3, 4, 5), value), np.eye(4))
            with self.assertRaises(ValueError):
                checked_array(img, [0, 1, 2, 4])

    def test_special_seg_name_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "BraTS20_Training_355"
            case.mkdir()
            p = case / "W39_1998.09.19_Segm.nii"
            p.touch()
            self.assertEqual(find_raw_file(tmp, case.name, "seg"), p)
            (case / "other_seg.nii.gz").touch()
            with self.assertRaises(ValueError):
                find_raw_file(tmp, case.name, "seg")


class CliTests(unittest.TestCase):
    def run_module(self, module, *args, success=True):
        process = subprocess.run([sys.executable, "-m", module, *map(str, args)],
                                 text=True, capture_output=True)
        if success:
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        else:
            self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        return process

    def test_full_pipeline_and_reject_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, pred, split, ids = create_fixture(root)
            common = ["--raw-dir", raw, "--pred-dir", pred, "--splits", split]
            self.run_module("src.eval.evaluate", *common, "--check-only")
            out = root / "eval"
            self.run_module("src.eval.evaluate", *common, "--out-dir", out)
            manifest = json.loads((out / "run_manifest.json").read_text())
            self.assertEqual(manifest["scope"], "full_split")
            with (out / "per_case.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 6)
            self.assertEqual(float(rows[0]["dice"]), 1.)
            viz = root / "viz"
            self.run_module("src.viz.overlay", "--raw-dir", raw, "--pred-dir", pred,
                            "--metrics-csv", out / "per_case.csv", "--out-dir", viz)
            self.assertEqual(len(list(viz.glob("*.png"))), 6)
            records = json.loads((viz / "selection.json").read_text())
            self.assertEqual(len({r["case_id"] for r in records}), 6)
            self.assertTrue(all(r["ras_axial_slice"] == 4 for r in records))
            (pred / f"{ids[0]}_pred_wt.nii.gz").unlink()
            bad_out = root / "must_not_exist"
            self.run_module("src.eval.evaluate", *common, "--out-dir", bad_out, success=False)
            self.assertFalse(bad_out.exists())

    def test_subset_and_unknown_unit_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, pred, split, ids = create_fixture(tmp)
            p = pred / f"{ids[0]}_pred_wt.nii.gz"
            img = nib.load(p)
            copied = nib.Nifti1Image(np.asarray(img.dataobj), img.affine)
            nib.save(copied, p)  # Trình ghi chưa đặt đơn vị: unit là unknown.
            out = Path(tmp) / "subset"
            common = ["--raw-dir", raw, "--pred-dir", pred, "--splits", split,
                      "--case-id", ids[0], "--out-dir", out]
            self.run_module("src.eval.evaluate", *common, success=False)
            self.assertFalse(out.exists())
            self.run_module("src.eval.evaluate", *common, "--assume-unknown-mm")
            manifest = json.loads((out / "run_manifest.json").read_text())
            self.assertEqual(manifest["scope"], "subset_smoke_test")
            self.assertTrue(manifest["assume_unknown_mm"])
            self.assertEqual(manifest["inputs"][0]["pred_unit"], "unknown")


class PrepareDataTests(unittest.TestCase):
    def test_discovery_and_file_audit(self):
        from src.eval.prepare_data import discover_raw_dir, audit_inputs
        with tempfile.TemporaryDirectory() as tmp:
            raw, pred, _, ids = create_fixture(tmp)
            self.assertEqual(discover_raw_dir(Path(tmp)), raw)
            self.assertEqual(len(audit_inputs(raw, pred, ids)), 6)

    def test_ambiguous_training_roots_rejected(self):
        from src.eval.prepare_data import discover_raw_dir
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_fixture(root / "first")
            create_fixture(root / "second")
            with self.assertRaises(ValueError):
                discover_raw_dir(root)

    def test_missing_flair_rejected(self):
        from src.eval.prepare_data import audit_inputs
        with tempfile.TemporaryDirectory() as tmp:
            raw, pred, _, ids = create_fixture(tmp)
            (raw / ids[0] / f"{ids[0]}_flair.nii.gz").unlink()
            with self.assertRaises(ValueError):
                audit_inputs(raw, pred, ids)

    def test_prepare_cli_55_cases(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, pred, split, ids = create_fixture(root)
            first = ids[0]
            for i in range(7, 56):
                cid = f"BraTS20_Training_{i:03d}"
                ids.append(cid)
                case = raw / cid
                case.mkdir()
                for ch in ("seg", "flair"):
                    shutil.copy2(raw / first / f"{first}_{ch}.nii.gz", case / f"{cid}_{ch}.nii.gz")
                shutil.copy2(pred / f"{first}_pred_wt.nii.gz", pred / f"{cid}_pred_wt.nii.gz")
            split.write_text(json.dumps({"counts": {"test": 55}, "splits": {"test": ids}}))
            out = root / "input_checks"
            process = subprocess.run([sys.executable, "-m", "src.eval.prepare_data",
                                      "--download-dir", str(raw), "--pred-dir", str(pred),
                                      "--splits", str(split), "--out-dir", str(out)],
                                     text=True, capture_output=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            record = json.loads((out / "inputs_checked.json").read_text(encoding="utf-8"))
            self.assertEqual(record["test_count"], 55)
            self.assertFalse(record["geometry_checked_all_cases"])
            self.assertFalse((out / "per_case.csv").exists())


if __name__ == "__main__":
    unittest.main()
