#!/usr/bin/env python3
#
#            _ _           _            _
#   ___   __| (_)_ __  ___| | ___   ___| | __
#  / _ \ / _` | | '_ \/ __| |/ _ \ / __| |/ /
# | (_) | (_| | | | | \__ \ | (_) | (__|   <
#  \___/ \__,_|_|_| |_|___/_|\___/ \___|_|\_\
#

'''
Script to encrypt and decrypt files/directories
'''

import argparse
import shutil
import subprocess
import sys
import tarfile
from collections import deque
from contextlib import contextmanager, suppress
from enum import IntEnum
from functools import partial
from itertools import compress, filterfalse, tee, islice
from operator import methodcaller, not_
from pathlib import Path
from typing import (
    Iterable,
    NoReturn,
    Sequence,
    TypeVar,
    Callable,
    Protocol,
    Generator,
)


# ============= #
#  Error Codes  #
# ============= #


class ExitCode(IntEnum):
    FILE_EXISTS = 1
    EXE_NOT_FOUND = 2
    FILE_NOT_FOUND = 3
    CMD_FAILED = 4


# ======================================= #
#  3rd party Encryption Decryption tools  #
# ======================================= #


class EDT(Protocol):
    def encrypt(self) -> None: ...
    def decrypt(self) -> None: ...


class OpenSSL:
    executable = 'openssl'
    __args = [
        'enc',
        '-aes-256-cbc',
        '-salt',
        '-pbkdf2',
        '-iter',
        '1000000',
        '-md',
        'sha512',
        '-base64',
    ]

    def __init__(self, in_file: Path) -> None:
        self.in_file = in_file

    def _cmd(self) -> list[str]:
        return [OpenSSL.executable] + OpenSSL.__args + self._get_in_file_arg()

    def _encrypt_cmd(self) -> list[str]:
        return self._cmd() + ['-e']

    def _decrypt_cmd(self) -> list[str]:
        return self._cmd() + ['-d']

    def _get_in_file_arg(self) -> list[str]:
        return ['-in', str(self.in_file)]

    def _get_out_file_arg(self, out_file: str | Path) -> list[str]:
        return ['-out', str(out_file)]

    def encrypt(self, out_file: str | Path) -> None:
        exec_cmd(
            self._encrypt_cmd() + self._get_out_file_arg(out_file),
            'Encryption failed',
        )

    def decrypt(self, out_file: str | Path) -> None:
        exec_cmd(
            self._decrypt_cmd() + self._get_out_file_arg(out_file), 'Decryption failed'
        )


# =================== #
#  Utility functions  #
# =================== #


A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


def partition(
    predicate: Callable[[A], bool], iterable: Iterable[A]
) -> tuple[Iterable[A], Iterable[A]]:
    t1, t2, p = tee(iterable, 3)
    p1, p2 = tee(map(predicate, p))
    return (compress(t1, map(not_, p1)), compress(t2, p2))


def compose(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    def inner(x: A) -> C:
        return f(g(x))

    return inner


def consume(iterator: Iterable, n: int | None = None) -> None:
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)


# ===================================== #


def exec_cmd(cmd: list[str], failure_message: str) -> NoReturn | None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        bail(failure_message, ExitCode.CMD_FAILED)


def to_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def print_prefix(banner: str, it: Iterable[Path]) -> None:
    for i in it:
        print(banner, i)


def bail(message: str, code: ExitCode) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(code.value)


def copy(source: Path, destination: Path) -> None:
    try:
        if source.is_dir():
            shutil.copytree(source, destination / source.stem)
        else:
            shutil.copy2(source, destination)
    except FileNotFoundError:
        bail(f'File {source} not found', ExitCode.FILE_NOT_FOUND)


def create_dir(dir: Path) -> NoReturn | None:
    try:
        dir.mkdir()
    except FileExistsError:
        bail(f'{dir} already exists', ExitCode.FILE_EXISTS)


def file2dir(filepath: Path) -> Path:
    return filepath.parent / file2stem(filepath)


def file2stem(filepath: Path) -> str:
    return filepath.stem.partition('.')[0]


def verify_executable_exists(executable: str) -> NoReturn | None:
    if not shutil.which(executable):
        bail(
            f'Executable {executable} not found in PATH',
            ExitCode.EXE_NOT_FOUND,
        )


@contextmanager
def stage_n_cleanup(
    input_files: list[Path], output_file: Path
) -> Generator[Path, None, None]:
    is_present = methodcaller('exists')
    to_dir = compose(file2dir, to_path)
    output_dir = to_dir(output_file)
    copy2output = partial(copy, destination=output_dir)
    input_paths1, input_paths2 = tee(map(to_path, input_files))
    missing_paths = filterfalse(is_present, input_paths1)
    existing_paths = filter(is_present, input_paths2)
    print_prefix('Missing path:', missing_paths)
    # in case failure occurs before this variable is binded again
    compressed_output = output_dir

    try:
        create_dir(output_dir)
        consume(map(copy2output, existing_paths))
        compressed_output = archive(output_file, output_dir)
        yield compressed_output
    finally:
        cleanup(output_dir, compressed_output)


def archive(compress_name: Path, directory: Path) -> Path:
    try:
        output_str = shutil.make_archive(
            str(compress_name),
            'gztar',
            root_dir=directory.parent,
            base_dir=directory.name,
        )
        return to_path(output_str)
    except FileNotFoundError:
        bail(f'Directory {directory} not found', ExitCode.FILE_NOT_FOUND)


def unarchive(filepath: Path):
    try:
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(filter='data')
    except FileNotFoundError:
        bail(f'File {filepath} not found', ExitCode.FILE_NOT_FOUND)


def unarchive_n_cleanup(filepath: Path) -> None:
    try:
        unarchive(filepath)
    finally:
        cleanup(filepath)


def encrypt(args: argparse.Namespace) -> None:
    with stage_n_cleanup(args.input, args.output) as compressed_output:
        encrypted_output = compressed_output.with_suffix(
            compressed_output.suffix + '.enc'
        )
        openssl = OpenSSL(compressed_output)
        openssl.encrypt(encrypted_output)


def decrypt(args: argparse.Namespace) -> None:
    encrypted_file = to_path(args.input)
    decrypted_file = encrypted_file.parent / encrypted_file.stem
    openssl = OpenSSL(encrypted_file)
    openssl.decrypt(decrypted_file)
    unarchive_n_cleanup(decrypted_file)


def cleanup(*files: Path) -> None:
    for file in files:
        with suppress(FileNotFoundError):
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description='After the Allfather, symbolizing wisdom and secrecy',
        epilog='No gaze shall graze your secrets',
    )
    subparser = parser.add_subparsers(required=True)

    enc_parser = subparser.add_parser('encrypt')
    enc_parser.add_argument(
        '-i',
        '--input',
        type=Path,
        nargs='+',
        help='Files to be encrypted',
        required=True,
    )
    enc_parser.add_argument(
        '-o',
        '--output',
        type=Path,
        help='Files to be encrypted',
        default=Path('sentinel'),
    )
    enc_parser.set_defaults(func=encrypt)

    dec_parser = subparser.add_parser('decrypt')
    dec_parser.add_argument(
        '-i', '--input', type=Path, help='File to be decrypted', required=True
    )
    dec_parser.set_defaults(func=decrypt)
    args = parser.parse_args(argv)
    args.func(args)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
