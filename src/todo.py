#!/usr/bin/env python3
#
#  _____         _
# |_   _|__   __| | ___
#   | |/ _ \ / _` |/ _ \
#   | | (_) | (_| | (_) |
#   |_|\___/ \__,_|\___/
#

"""
Script to interact with Todoist
"""

import argparse
from http.client import HTTPResponse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import NamedTuple, NoReturn, Sequence, TypeVar, Callable


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                          Types                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


@dataclass(frozen=True)
class Project:
    id: int
    name: str


class TodoState(NamedTuple):
    creds: Path
    projects: Path
    todos: Path


class XdgDirs(StrEnum):
    DATA = 'XDG_DATA_HOME'
    STATE = 'XDG_STATE_HOME'


class ExitCode(IntEnum):
    CREDS_NOT_FOUND = 1
    REQUEST_FAILED = 2
    PROJECTS_NOT_FOUND = 3
    PROJECTS_SAVE_FAILED = 4


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    Utility Functions                     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


def bail(msg: str, code: ExitCode) -> NoReturn:
    print(msg, file=sys.stderr)
    exit(code.value)


def to_path(file: str) -> Path:
    return Path(file).expanduser().resolve()


def get_path(xdg_name: XdgDirs) -> Path:
    match xdg_name:
        case XdgDirs.DATA:
            return to_path(os.getenv('XDG_DATA_HOME', '~/.local/share'))
        case XdgDirs.STATE:
            return to_path(os.getenv('XDG_STATE_HOME', '~/.local/state'))


def compose2(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    def inner(x: A) -> C:
        return f(g(x))

    return inner


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                         Globals                          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


todoState = TodoState(
    creds=get_path(XdgDirs.DATA) / 'todoist' / 'creds.json',
    projects=get_path(XdgDirs.STATE) / 'todoist' / 'projects.json',
    todos=get_path(XdgDirs.STATE) / 'todoist' / 'todos.json',
)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                   Core Implementation                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def get_token() -> str:
    try:
        with open(todoState.creds) as f:
            return json.load(f)['token']
    except Exception as e:
        bail(f'ERROR: Todoist Creds not found, REASON: {e}', ExitCode.CREDS_NOT_FOUND)


def request(
    url: str, data: dict | None = None, headers: dict[str, str] = {}
) -> HTTPResponse:
    default_headers = {
        'Content-Type': 'application/json',
        'X-Request-Id': f'{uuid.uuid4()}',
        'Authorization': f'Bearer {get_token()}',
    }
    req = urllib.request.Request(
        url,
        headers=default_headers | headers,
        data=data and json.dumps(data).encode(),
    )
    try:
        res = urllib.request.urlopen(req, timeout=5)
        if res.status != 200:
            raise Exception(
                f'Todoist responded with code: {res.status}, content: {res.read().decode('utf-8')}'
            )
        return res
    except urllib.error.URLError as e:
        bail(f'ERROR: Request failed {url}, REASON: {e}', ExitCode.REQUEST_FAILED)


def dict2Project(d: dict) -> Project:
    return Project(d['id'], d['name'])


def download(file: Path, res: HTTPResponse) -> None:
    with open(file, 'wb') as f:
        # copy 16 KB chunk
        chunk_size = 1024 * 16
        with res as reader:
            while chunk := reader.read(chunk_size):
                f.write(chunk)


def get_projects() -> list[Project]:
    # if not todoState.projects.exists():
    todoState.projects.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(todoState.projects, 'wb') as f:
            download(
                todoState.projects, request('https://api.todoist.com/rest/v2/projects')
            )
    except Exception as e:
        bail(
            f'ERROR: Failed to save project state, REASON: {e}',
            ExitCode.PROJECTS_SAVE_FAILED,
        )
    try:
        with open(todoState.projects) as f:
            return json.load(f, object_hook=dict2Project)
    except Exception as e:
        bail(
            f'ERROR: Failed to read projects, REASON: {e}', ExitCode.PROJECTS_NOT_FOUND
        )


def display_projects() -> None:
    for p in get_projects():
        print(f'- {p.name}')


def get_controller(args: argparse.Namespace):
    if args.projects:
        display_projects()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='A client to interact with Todoist', epilog='Happy todoing'
    )
    subparser = parser.add_subparsers(required=True)

    # -- get sub-command
    get_parser = subparser.add_parser('get')
    get_group_parser = get_parser.add_mutually_exclusive_group(required=True)
    get_group_parser.add_argument(
        '-p', '--projects', action='store_true', help='Get all projects'
    )
    get_parser.set_defaults(func=get_controller)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    pass
    raise SystemExit(main())
