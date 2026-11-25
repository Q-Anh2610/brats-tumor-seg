"""Vẽ MRI và WT ground truth trực tiếp từ NPZ BraTS-Africa đã xử lý."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


MRI_CMAP = plt.colormaps['gray'].with_extremes(bad='black')
WT_CMAP = ListedColormap(['#ff4d4f'])


def draw_group(data_dir: Path, group: str, rows: pd.DataFrame, out_dir: Path) -> list[dict]:
    """Chọn ba ca theo thể tích WT gốc, vẽ cùng một lát cho bốn MRI và mask."""
    ordered = rows.sort_values(['wt_volume_ml', 'case_id']).reset_index(drop=True)
    positions = [int((len(ordered) - 1) * fraction) for fraction in (0.1, 0.5, 0.9)]
    selected = []
    fig, axes = plt.subplots(3, 5, figsize=(14, 9), layout='constrained')
    for line, (position, tag) in enumerate(zip(positions, ['Small', 'Median', 'Large'])):
        row = ordered.iloc[position]
        case_id = row['case_id']
        with np.load(data_dir / 'cases' / f'{case_id}.npz', allow_pickle=False) as archive:
            masks = archive['masks']
            z = int(masks.sum(axis=(1, 2)).argmax())
            images = archive['images'][z].astype(np.float32)
            mask = masks[z]
        for channel, name in enumerate(['T1', 'T1ce', 'T2', 'FLAIR']):
            axis = axes[line, channel]
            foreground = images[channel][images[channel] != 0]
            low, high = np.percentile(foreground, [1, 99])
            shown = np.ma.masked_where(images[channel].T == 0, images[channel].T)
            axis.imshow(shown, cmap=MRI_CMAP, origin='lower', vmin=low, vmax=high)
            if line == 0:
                axis.set_title(name)
        flair = np.ma.masked_where(images[3].T == 0, images[3].T)
        axes[line, 4].imshow(flair, cmap=MRI_CMAP, origin='lower',
                             vmin=np.percentile(images[3][images[3] != 0], 1),
                             vmax=np.percentile(images[3][images[3] != 0], 99))
        axes[line, 4].imshow(np.ma.masked_where(mask.T == 0, mask.T),
                             cmap=WT_CMAP, alpha=0.45, origin='lower', vmin=0, vmax=1)
        if line == 0:
            axes[line, 4].set_title('FLAIR + WT ground truth')
        axes[line, 0].set_ylabel(f'{tag}, {row["wt_volume_ml"]:.1f} mL\n{case_id}\n224x224, z={z}')
        for axis in axes[line]:
            axis.set_xticks([])
            axis.set_yticks([])
        selected.append({'case_id': case_id, 'group': group, 'selection': tag,
                         'wt_volume_ml_original': float(row['wt_volume_ml']), 'slice_z': z})
    fig.suptitle(f'BraTS-Africa, {group}, preprocessed MRI and WT ground truth')
    fig.savefig(out_dir / f'africa_npz_{group}.png', dpi=160)
    plt.close(fig)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.data_dir / 'eda_africa_cases.csv')
    index = json.loads((args.data_dir / 'index.json').read_text(encoding='utf-8'))
    if len(frame) != 146 or frame.case_id.nunique() != 146:
        raise ValueError('EDA cần đủ 146 ca không trùng')
    if set(frame.case_id) != {row['case_id'] for row in index}:
        raise ValueError('ID trong EDA không khớp index')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selection = []
    for group, expected in [('95_Glioma', 95), ('51_OtherNeoplasms', 51)]:
        rows = frame.loc[frame['group'] == group]
        if len(rows) != expected:
            raise ValueError(f'{group}: số ca không đúng')
        selection.extend(draw_group(args.data_dir, group, rows, args.out_dir))
    (args.out_dir / 'selection.json').write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Đã lưu hai hình MRI và WT, tổng cộng 6 ca tại', args.out_dir)


if __name__ == '__main__':
    main()
