"""Kiểm tra dữ liệu đầu vào và việc giữ Africa ngoài bước fit PCA."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess.compare_africa import CHANNELS, MRI_FEATURES, load_features, run_pca


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({
            'case_id': [f'case_{i}' for i in range(515)],
            'brain_voxels': rng.integers(900000, 1500000, 515),
            'wt_voxels': np.full(515, 10000), 'wt_volume_ml': np.full(515, 10.0),
            'num_slices_z': np.full(515, 155), 'num_tumor_slices': np.full(515, 50),
            'wt_ratio_in_brain': np.full(515, 0.01), 'tumor_slice_ratio': np.full(515, 50 / 155),
            'spacing_x_mm': np.ones(515), 'spacing_y_mm': np.ones(515), 'spacing_z_mm': np.ones(515)
        })
        for channel, _ in CHANNELS:
            for stat in ('mean', 'std', 'p05', 'p50', 'p95'):
                frame[f'{channel}_{stat}_brain'] = rng.uniform(1, 100, 515)
        self.reference = frame.iloc[:369].copy()
        self.africa = frame.iloc[369:].copy()
        self.africa['group'] = ['95_Glioma'] * 95 + ['51_OtherNeoplasms'] * 51
        ids = self.reference.case_id.tolist()
        self.splits = {'train': ids[:259], 'val': ids[259:314], 'test': ids[314:]}

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        self.reference.to_csv(self.root / 'reference.csv', index=False)
        self.africa.to_csv(self.root / 'africa.csv', index=False)
        (self.root / 'splits.json').write_text(json.dumps({'splits': self.splits}), encoding='utf-8')
        return load_features(self.root / 'reference.csv', self.root / 'africa.csv', self.root / 'splits.json')

    def test_split_overlap_rejected(self):
        self.splits['test'][0] = self.splits['train'][0]
        with self.assertRaisesRegex(ValueError, 'trùng'):
            self.load()

    def test_wrong_cohort_rejected(self):
        self.africa.iloc[0, self.africa.columns.get_loc('group')] = 'unknown'
        with self.assertRaisesRegex(ValueError, '95 glioma'):
            self.load()

    def test_nonfinite_feature_rejected(self):
        self.africa.iloc[0, self.africa.columns.get_loc('t1_mean_brain')] = np.nan
        with self.assertRaisesRegex(ValueError, 'NaN'):
            self.load()

    def test_pca_fit_unchanged_when_external_values_change(self):
        frame = self.load()
        first = self.root / 'first'
        second = self.root / 'second'
        for directory in (first, second):
            (directory / 'figures').mkdir(parents=True)
        before = run_pca(frame, MRI_FEATURES, first, 'mri', 'test')
        changed = frame.copy()
        heldout = changed['split'] != 'train'
        changed.loc[heldout, MRI_FEATURES] *= 1000
        after = run_pca(changed, MRI_FEATURES, second, 'mri', 'test')
        for key in ['scaler_mean', 'scaler_scale', 'pca_mean', 'components', 'explained_variance_ratio']:
            np.testing.assert_allclose(before[key], after[key], rtol=0, atol=0)
        self.assertEqual(set(before['fit_cases']), set(self.splits['train']))


if __name__ == '__main__':
    unittest.main()
