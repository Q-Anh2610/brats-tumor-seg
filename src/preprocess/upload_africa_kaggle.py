"""Đưa bộ NPZ đã kiểm tra lên Kaggle, mặc định tạo dataset riêng tư."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--validation-report', type=Path, required=True)
    parser.add_argument('--dataset-card', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.validation_report.read_text(encoding='utf-8'))
    index = json.loads((args.data_dir / 'index.json').read_text(encoding='utf-8'))
    if report['num_valid_cases'] != 146 or len(index) != 146 or len(report['records']) != 146:
        raise ValueError('Chỉ upload sau khi đủ 146 ca đã kiểm tra')
    expected = {row['case_id'] for row in report['records']}
    if expected != {row['case_id'] for row in index}:
        raise ValueError('Báo cáo và index không khớp')
    from src.preprocess.validate_africa_dataset import file_hash
    for row in report['records']:
        case_id = row['case_id']
        for folder, extension, key in [('cases', 'npz', 'npz_sha256'),
                                       ('metadata', 'json', 'metadata_sha256')]:
            path = args.data_dir / folder / f'{case_id}.{extension}'
            if file_hash(path) != row[key]:
                raise ValueError(f'{case_id}: file thay đổi sau khi kiểm tra')
    shutil.copyfile(args.metadata, args.data_dir / 'dataset-metadata.json')
    shutil.copyfile(args.validation_report, args.data_dir / 'validation_report.json')
    shutil.copyfile(args.dataset_card, args.data_dir / 'README.txt')
    # Không truyền --public, Kaggle sẽ tạo dataset riêng tư.
    subprocess.run([sys.executable, '-m', 'kaggle', 'datasets', 'create',
                    '-p', str(args.data_dir.resolve()), '--dir-mode', 'tar'], check=True)


if __name__ == '__main__':
    main()
