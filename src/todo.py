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
from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import NamedTuple, NoReturn, Sequence, TypeVar, Callable


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                          Types                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


@dataclass(frozen=True)
class Due:
    string: str
    date: date
    is_recurring: bool
    datetime: datetime | None
    timezone: ZoneInfo | None


class Priority(IntEnum):
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4


class Label(StrEnum):
    CHORE = 'chore'
    WORK = 'work'
    AUTO = 'automation'
    MOVIE = 'movie'
    LEARN = 'learn'


@dataclass(frozen=True)
class Project:
    id: int
    name: str


@dataclass(frozen=True)
class Task:
    id: int
    content: str
    description: str
    due: Due | None
    labels: list[Label]
    priority: Priority
    project_id: int
    section_id: int | None
    parent_id: int | None
    url: str


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
    PROJECT_NOT_FOUND = 3
    TASK_NOT_FOUND = 4


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


def dict2Task(d: dict) -> Task:
    due = d.get('due')
    return Task(
        id=d['id'],
        content=d['content'],
        description=d['description'],
        due=due
        and Due(
            due['string'],
            due.get('date') and date.fromisoformat(due['date']),
            due['is_recurring'],
            due.get('datetime') and datetime.fromisoformat(due['datetime']),
            due.get('timezone') and ZoneInfo(due['timezone']),
        ),
        labels=[Label(l) for l in d['labels']],
        priority=Priority(d['priority']),
        project_id=d['project_id'],
        section_id=d.get('section_id'),
        parent_id=d.get('parent_id'),
        url=d['url'],
    )


def download(file: Path, res: HTTPResponse) -> None:
    with open(file, 'wb') as f:
        # copy 16 KB chunk
        chunk_size = 1024 * 16
        with res as reader:
            while chunk := reader.read(chunk_size):
                f.write(chunk)


def download_object(url: str, state_file: Path, exit_code: ExitCode) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        download(state_file, request(url))
    except Exception as e:
        bail(
            f'ERROR: Failed to save {state_file.name} state, REASON: {e}',
            exit_code,
        )


def get_object(
    url: str,
    state_file: Path,
    exit_code: ExitCode,
    refresh: bool = False,
) -> dict:
    if not state_file.exists() or refresh:
        download_object(url, state_file, exit_code)
    try:
        with open(state_file) as f:
            return json.load(f)
    except Exception as e:
        bail(f'ERROR: Failed to read {state_file.name}, REASON: {e}', exit_code)


def get_projects() -> list[Project]:
    return [
        dict2Project(p)
        for p in get_object(
            'https://api.todoist.com/rest/v2/projects',
            todoState.projects,
            ExitCode.PROJECT_NOT_FOUND,
        )
    ]


def get_active_tasks() -> list[Task]:
    return [
        dict2Task(t)
        for t in get_object(
            'https://api.todoist.com/rest/v2/tasks',
            todoState.todos,
            ExitCode.TASK_NOT_FOUND,
        )
    ]


def display_projects() -> None:
    for p in get_projects():
        print(f'- {p.name}')


def display_active_tasks() -> None:
    for t in get_active_tasks():
        print(f'- {t.content}')


def get_controller(args: argparse.Namespace):
    if args.projects:
        display_projects()
    elif args.active_tasks:
        display_active_tasks()


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
    get_group_parser.add_argument(
        '-a', '--active-tasks', action='store_true', help='Get all active tasks'
    )
    get_parser.set_defaults(func=get_controller)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    pass
    raise SystemExit(main())
