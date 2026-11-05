#!/usr/bin/env python3

import argparse
import os
import signal
import sys


def check_root() -> None:
    if os.getuid() != 0:
        print('Error: this script must be run as root.')
        sys.exit(1)


def ghost_file(arg: argparse.Namespace) -> None:
    filepath = arg.ghost_file
    if not os.path.exists(filepath):
        print(f'Error: {filepath} does not exist')
        return

    fd = os.open(filepath, os.O_RDWR)
    os.unlink(filepath)
    pid = os.fork()

    if pid > 0:
        print(f'File "{filepath}" is now ghosted.')
        print(f'Background Process PID: {pid}')
        print(f'Use: `sudo python3 ghost_file.py hunt {pid}` to find it.')
        sys.exit(0)

    os.setsid()
    try:
        signal.pause()
    finally:
        os.close(fd)


def hunt_ghost(arg: argparse.Namespace):
    pid = arg.pid
    fd_path = f"/proc/{pid}/fd"
    found = False
    try:
        for fd in os.listdir(fd_path):
            full_path = os.path.join(fd_path, fd)
            try:
                link = os.readlink(full_path)
                if "(deleted)" in link:
                    print(f'[+] Ghost found!')
                    print(f'    Path to data: {full_path}')
                    print(f'    Original name: {link}')
                    found = True
            except OSError:
                continue
        if not found:
            print(f'No ghost files found for PID {pid}.')
    except Exception as e:
        print(f'Error accessing PID {pid}: {e}')


def search_all_ghosts(arg: argparse.Namespace):
    print('Searching all processes for ghost files...')
    for entry in os.listdir('/proc'):
        if entry.isdigit():
            arg.pid = entry
            hunt_ghost(arg)


def main():
    check_root()

    parser = argparse.ArgumentParser(description='Ghost a file')
    subparser = parser.add_subparsers(required=True)

    ghost_parser = subparser.add_parser('ghost')
    ghost_parser.add_argument('ghost_file', help='Hide a file')
    ghost_parser.set_defaults(func=ghost_file)

    hunt_parser = subparser.add_parser('hunt')
    hunt_parser.add_argument('pid', help='Find a ghost by PID')
    hunt_parser.set_defaults(func=hunt_ghost)

    search_parser = subparser.add_parser('search')
    search_parser.set_defaults(func=search_all_ghosts)

    args = parser.parse_args()
    args.func(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
