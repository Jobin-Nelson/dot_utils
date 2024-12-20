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
import urllib.parse
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict, field, fields
from enum import IntEnum, StrEnum
from datetime import date, datetime
from typing import NamedTuple, NoReturn, Sequence, Optional


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                          Types                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


@dataclass(kw_only=True)
class Due:
    string: str
    date: Optional[date] = None
    is_recurring: bool = False
    datetime: Optional[datetime] = None

    def __post__init__(self):
        self.date = date and date.fromisoformat(date)
        self.datetime = datetime and datetime.fromisoformat(datetime)


DUE_FIELDS = [f.name for f in fields(Due)]


class Priority(IntEnum):
    P1 = 4
    P2 = 3
    P3 = 2
    P4 = 1


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


PROJECT_FIELDS = [f.name for f in fields(Project)]


@dataclass(kw_only=True)
class Task:
    content: str
    project_id: int
    description: str = ''
    id: Optional[int] = None
    priority: Priority = Priority.P4
    labels: list[Label] = field(default_factory=list)
    url: Optional[str] = None
    section_id: Optional[int] = None
    parent_id: Optional[int] = None
    due: Optional[Due] = None

    def __post_init__(self):
        self.due = self.due and Due(
            **{f: v for f, v in self.due.items() if f in DUE_FIELDS}
        )
        self.priority = Priority(self.priority)
        self.labels = [Label(l) for l in self.labels]


TASK_FIELDS = [f.name for f in fields(Task)]


class TodoState(NamedTuple):
    creds: Path
    projects: Path
    tasks: Path


class XdgDirs(StrEnum):
    DATA = 'XDG_DATA_HOME'
    STATE = 'XDG_STATE_HOME'


class ExitCode(IntEnum):
    CREDS_NOT_FOUND = 1
    REQUEST_FAILED = 2
    PROJECT_NOT_FOUND = 3
    TASK_NOT_FOUND = 4
    LABEL_NOT_FOUND = 4


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    Utility Functions                     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


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


TODO_STATE = TodoState(
    creds=get_path(XdgDirs.DATA) / 'todoist' / 'creds.json',
    projects=get_path(XdgDirs.STATE) / 'todoist' / 'projects.json',
    tasks=get_path(XdgDirs.STATE) / 'todoist' / 'tasks.json',
)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                   Core Implementation                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def get_token() -> str:
    try:
        with open(TODO_STATE.creds) as f:
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
        data=data and json.dumps(data).encode('utf-8'),
        method='POST' if data else 'GET',
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
    return Project(**{f: v for f, v in d.items() if f in PROJECT_FIELDS})


def dict2Task(d: dict) -> Task:
    return Task(**{f: v for f, v in d.items() if f in TASK_FIELDS})


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


def upload_object(url: str, data: dict, state_file: Path, exit_code: ExitCode) -> None:
    request(url, data)
    download_object(url, state_file, exit_code)


def get_projects() -> list[Project]:
    return [
        dict2Project(p)
        for p in get_object(
            'https://api.todoist.com/rest/v2/projects',
            TODO_STATE.projects,
            ExitCode.PROJECT_NOT_FOUND,
        )
    ]


def get_active_tasks() -> list[Task]:
    return [
        dict2Task(t)
        for t in get_object(
            'https://api.todoist.com/rest/v2/tasks',
            TODO_STATE.tasks,
            ExitCode.TASK_NOT_FOUND,
        )
    ]


def add_task(task: Task) -> None:
    # $ curl "https://api.todoist.com/rest/v2/tasks" \
    #     --data '{"content": "Buy Milk", "due_string": "tomorrow at 12:00", "due_lang": "en", "priority": 4}' \
    task_dict = {k: v for k, v in asdict(task).items() if v}
    due_dict = (
        {'due_' + k: v for k, v in task_dict.pop('due').items() if v}
        if task_dict.get('due')
        else {}
    )
    print(json.dumps(task_dict | due_dict))
    upload_object(
        "https://api.todoist.com/rest/v2/tasks",
        task_dict | due_dict,
        TODO_STATE.tasks,
        ExitCode.TASK_NOT_FOUND,
    )


def display_projects() -> None:
    for p in get_projects():
        print(f'- {p.name}')


def display_active_tasks() -> None:
    for t in get_active_tasks():
        print(f'- {t.content}')


def get_controller(args: argparse.Namespace) -> None:
    if args.projects:
        display_projects()
    elif args.active_tasks:
        display_active_tasks()


def add_task_controller(args: argparse.Namespace) -> None:
    # validate section
    due_field = args.due_string or args.due_date or args.due_datetime
    add_task(
        Task(
            content=args.content,
            description=args.description,
            project_id=[p.id for p in get_projects() if p.name == args.project][0],
            priority=args.priority,
            section_id=args.section,
            labels=args.label,
            due=due_field
            and Due(
                string=due_field if args.due_string else due_field.strftime('%d %b'),
                date=args.due_date,
                datetime=args.due_datetime,
            ),
        )
    )


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                  Command Line Options                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='A client to interact with Todoist', epilog='Happy todoing'
    )
    subparser = parser.add_subparsers(required=True)

    # -- get sub-command
    get_parser = subparser.add_parser('get', help='Get a Todoist object')
    get_group_parser = get_parser.add_mutually_exclusive_group(required=True)
    get_group_parser.add_argument(
        '-p', '--projects', action='store_true', help='Get all projects'
    )
    get_group_parser.add_argument(
        '-a', '--active-tasks', action='store_true', help='Get all active tasks'
    )
    get_parser.set_defaults(func=get_controller)

    # -- add sub-command
    add_parser = subparser.add_parser('add', help='Add a Todoist object')
    add_obj_parser = add_parser.add_subparsers(required=True)

    # -- add task sub-command
    add_task_parser = add_obj_parser.add_parser('task', help='Add a task')
    add_task_parser.add_argument(
        '-c', '--content', help='Content of task', required=True
    )
    add_task_parser.add_argument(
        '--description', help='Description of task', default=''
    )
    add_task_parser.add_argument(
        '--project',
        help='Add task to the project',
        default='Inbox',
        choices=[p.name for p in get_projects()],
    )
    add_task_parser.add_argument(
        '-s',
        '--section',
        help='Add task to the section',
    )
    add_task_parser.add_argument(
        '-l',
        '--label',
        action='append',
        type=Label,
        help='Add task with label',
        default=[],
        choices=[l.value for l in Label],
    )
    add_task_parser.add_argument(
        '-p',
        '--priority',
        type=Priority,
        help='Add task with priority',
        default=Priority.P4,
        choices=[p.value for p in Priority],
    )
    add_task_parser.add_argument(
        '-d',
        '--due-string',
        help='Task due by',
    )
    add_task_parser.add_argument(
        '--due-date',
        type=date.fromisoformat,
        help='Task due by date',
    )
    add_task_parser.add_argument(
        '--due-datetime',
        type=datetime.fromisoformat,
        help='Task due by datetime',
    )
    add_task_parser.set_defaults(func=add_task_controller)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
