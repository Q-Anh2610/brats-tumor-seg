"""Mở lại NPZ từ Kaggle, kiểm tra ảnh, WT và metadata mà không cần NIfTI gốc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_COUNTS = {'95_Glioma': 95, '51_OtherNeoplasms': 51}
CHANNEL_ORDER = ['T1', 'T1ce', 'T2', 'FLAIR']


def inspect_case(data_dir: Path, row: dict) -> dict:
    """Kiểm tra một ca và trả về các thông tin để lập bảng kết quả."""
    case_id = row['case_id']
    metadata = json.loads((data_dir / 'metadata' / f'{case_id}.json').read_text(encoding='utf-8'))
    if metadata['case_id'] != case_id or metadata['group'] != row['group']:
        raise ValueError(f'{case_id}: định danh hoặc nhóm không khớp metadata')
    if metadata['channel_order'] != CHANNEL_ORDER or metadata['wt_labels'] != [1, 2, 3]:
        raise ValueError(f'{case_id}: thứ tự kênh hoặc nhãn WT không đúng')
    z = metadata['num_slices_z']
    with np.load(data_dir / 'cases' / f'{case_id}.npz', allow_pickle=False) as archive:
        required = {'images', 'masks', 'slice_has_tumor', 'slice_has_brain'}
        if set(archive.files) != required:
            raise ValueError(f'{case_id}: sai các key trong NPZ')
        images, masks = archive['images'], archive['masks']
        if z != 155 or images.shape != (z, 4, 224, 224) or masks.shape != (z, 224, 224):
            raise ValueError(f'{case_id}: shape không đúng bản 224x224 v1.1')
        if images.dtype != np.float16 or masks.dtype != np.uint8:
            raise ValueError(f'{case_id}: dtype không đúng')
        if not np.isfinite(images).all() or not np.isin(masks, [0, 1]).all():
            raise ValueError(f'{case_id}: ảnh có NaN/Inf hoặc mask không nhị phân')
        tumor_flags = masks.reshape(z, -1).any(axis=1).astype(np.uint8)
        brain_flags = images.reshape(z, -1).any(axis=1).astype(np.uint8)
        for key, expected in [('slice_has_tumor', tumor_flags), ('slice_has_brain', brain_flags)]:
            if archive[key].dtype != np.uint8 or not np.array_equal(archive[key], expected):
                raise ValueError(f'{case_id}: {key} không khớp dữ liệu')
        if int(tumor_flags.sum()) != metadata['num_tumor_slices']:
            raise ValueError(f'{case_id}: số lát có u không khớp metadata')
        if int(brain_flags.sum()) != metadata['num_brain_slices'] or not masks.any():
            raise ValueError(f'{case_id}: số lát não sai hoặc mask WT rỗng')
        for key in ['num_slices_z', 'num_tumor_slices', 'num_brain_slices']:
            if row[key] != metadata[key]:
                raise ValueError(f'{case_id}: index và metadata không khớp {key}')
        result = {'case_id': case_id, 'group': row['group'], 'num_slices_z': z,
                  'num_tumor_slices': int(tumor_flags.sum()),
                  'num_brain_slices': int(brain_flags.sum()),
                  'wt_pixels_224': int(masks.sum()), 'valid': True}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.data_dir / 'index.json').read_text(encoding='utf-8'))
    ids = [row['case_id'] for row in index]
    if len(ids) != 146 or len(set(ids)) != 146:
        raise ValueError('Cần đủ 146 ID không trùng')
    for folder, extension in [('cases', 'npz'), ('metadata', 'json')]:
        if {p.stem for p in (args.data_dir / folder).glob(f'*.{extension}')} != set(ids):
            raise ValueError(f'Thiếu hoặc thừa tệp trong {folder}')
    cohorts = json.loads((args.data_dir / 'cohorts.json').read_text(encoding='utf-8'))
    for group, expected in GROUP_COUNTS.items():
        group_ids = {row['case_id'] for row in index if row['group'] == group}
        if len(group_ids) != expected or set(cohorts.get(group, [])) != group_ids:
            raise ValueError(f'{group}: số ca hoặc danh sách nhóm không đúng')
    results = []
    for number, row in enumerate(index, 1):
        results.append(inspect_case(args.data_dir, row))
        if number % 10 == 0 or number == len(index):
            print(f'Đã mở và kiểm tra {number}/{len(index)} ca', flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.out_dir / 'npz_check_cases.csv', index=False)
    summary = {'num_cases': len(results), 'num_valid_cases': len(results),
               'by_group': GROUP_COUNTS, 'image_shape': [155, 4, 224, 224],
               'mask_shape': [155, 224, 224], 'image_dtype': 'float16', 'mask_dtype': 'uint8',
               'note': 'Kiểm tra NPZ và metadata, không kiểm tra lại hình học bằng NIfTI gốc'}
    (args.out_dir / 'npz_check_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Hoàn tất kiểm tra 146 ca, kết quả tại', args.out_dir)


if __name__ == '__main__':
    main()
