"""Tải bản Africa đã xử lý, giải TAR và kiểm tra checksum từng ca."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from src.preprocess.validate_africa_dataset import file_hash


def unpack_archives(data_dir: Path) -> None:
    # Archive của Kaggle chứa tên file tương đối, cần giải vào đúng thư mục con.
    for folder in ('cases', 'metadata'):
        archive_path = data_dir / f'{folder}.tar'
        if archive_path.is_file():
            destination = data_dir / folder
            destination.mkdir(exist_ok=True)
            with tarfile.open(archive_path) as archive:
                archive.extractall(destination, filter='data')


def verify_download(data_dir: Path) -> int:
    report = json.loads((data_dir / 'validation_report.json').read_text(encoding='utf-8'))
    index = json.loads((data_dir / 'index.json').read_text(encoding='utf-8'))
    ids = {row['case_id'] for row in index}
    if len(index) != 146 or len(ids) != 146 or len(report['records']) != 146:
        raise ValueError('Cần đủ 146 ca không trùng')
    if ids != {row['case_id'] for row in report['records']}:
        raise ValueError('Index và validation_report không khớp')
    for folder, extension in [('cases', 'npz'), ('metadata', 'json')]:
        if {p.stem for p in (data_dir / folder).glob(f'*.{extension}')} != ids:
            raise ValueError(f'Thiếu hoặc thừa file trong {folder}')
    for row in report['records']:
        case_id = row['case_id']
        for folder, extension, key in [('cases', 'npz', 'npz_sha256'),
                                       ('metadata', 'json', 'metadata_sha256')]:
            if file_hash(data_dir / folder / f'{case_id}.{extension}') != row[key]:
                raise ValueError(f'{case_id}: checksum sai trong {folder}')
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--dataset', default='trinhthuhuyenhuyen/brats-africa-preprocessed-224-v11')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        try:
            count = verify_download(args.out_dir)
        except (OSError, ValueError, KeyError):
            print('Chưa có bộ dữ liệu hoàn chỉnh, bắt đầu tải từ Kaggle', flush=True)
        else:
            print(f'Đã có đủ {count} ca và checksum khớp, dùng lại dữ liệu trong runtime')
            return
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(args.dataset, path=str(args.out_dir), unzip=True, quiet=False)
        unpack_archives(args.out_dir)
    count = verify_download(args.out_dir)
    print(f'Đã kiểm tra checksum NPZ và metadata của {count} ca tại {args.out_dir}')


if __name__ == '__main__':
    main()
