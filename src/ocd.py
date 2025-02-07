#!/usr/bin/env python3
#
#  ██████╗  ██████╗██████╗
# ██╔═══██╗██╔════╝██╔══██╗
# ██║   ██║██║     ██║  ██║
# ██║   ██║██║     ██║  ██║
# ╚██████╔╝╚██████╗██████╔╝
#  ╚═════╝  ╚═════╝╚═════╝
#

"""Script to organize files"""

import argparse
import re
import sys
from datetime import datetime
from enum import IntEnum
from functools import partial
from operator import attrgetter, truediv
from pathlib import Path
from typing import NoReturn, Sequence, TypeVar, Callable


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                       Error Codes                        ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


class ExitCode(IntEnum):
    PATH_NOT_FOUND = 1


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                  Functional Utilities                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


def compose2(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    def inner(x: A) -> C:
        return f(g(x))

    return inner


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                   Core Implementation                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def bail(message: str, code: ExitCode) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(code.value)


def date2str(dt: datetime) -> str:
    return dt.strftime('%Y/%b').lower()


def match_year_month(filename: str) -> re.Match | None:
    return re.search(r'\d{4}-\d{2}', filename)


def match2date(match: re.Match) -> datetime:
    return datetime.strptime(match.group(), r'%Y-%m')


def create_parents(filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)


def rename(from_path: Path, to_path: Path) -> None:
    create_parents(to_path)
    from_path.rename(to_path)


def organize_file(file: Path) -> None:
    name = attrgetter('name')
    match_file = compose2(match_year_month, name)
    match2str = compose2(date2str, match2date)
    str2path = partial(truediv, file.parent)
    match2path = compose2(str2path, match2str)

    match = match_file(file)
    # If date cannot be parsed or is already organized we return
    if match is None or match2str(match) in str(file):
        return
    target_path = match2path(match) / name(file)
    rename(file, target_path)


def organise_dir(directory: Path) -> None:
    if not directory.is_dir():
        bail(f'ERROR: Path {directory} not found', ExitCode.PATH_NOT_FOUND)

    files = [file for file in directory.iterdir() if file.is_file()]
    for file in files:
        organize_file(file)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=Path, help='Directory to operate on')
    args = parser.parse_args(argv)
    organise_dir(args.dir)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
