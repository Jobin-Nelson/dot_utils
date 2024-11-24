#!/usr/bin/env python3

import argparse
from datetime import datetime
from pathlib import Path


def organise_dir(dir: Path):
    files = [file for file in dir.iterdir() if file.is_file()]
    files = [file for file in files if not file.name.startswith('2024-11')]
    [organize_file(file) for file in files]


def organize_file(file: Path):
    date = datetime.strptime(file.stem, r'%Y-%m-%d')
    target_dir = file.parent / date.strftime('%Y/%b')
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / file.name
    file.rename(target_file)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=Path, help='Directory to operate on')
    args = parser.parse_args()
    organise_dir(args.dir)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
