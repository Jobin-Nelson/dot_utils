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

from __future__ import annotations


import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, asdict, field, fields
from datetime import date, datetime
from enum import IntEnum, StrEnum
from functools import partial
from http.client import HTTPResponse
from pathlib import Path
from typing import NamedTuple, NoReturn, Sequence, Optional, TypeVar, Callable


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
        self.date = (
            date.fromisoformat(self.date) if isinstance(self.date, str) else self.date
        )
        self.datetime = (
            datetime.fromisoformat(self.datetime)
            if isinstance(self.datetime, str)
            else self.datetime
        )


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

    def __str__(self) -> str:
        return f'- {self.name}'


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
    is_completed: bool = False
    created_at: Optional[datetime] = None

    def __post_init__(self):
        self.due = (
            Due(**{f: v for f, v in self.due.items() if f in DUE_FIELDS})
            if isinstance(self.due, dict)
            else self.due
        )
        self.priority = Priority(self.priority)
        self.labels = [Label(l) for l in self.labels]
        self.created_at = (
            datetime.fromisoformat(self.created_at)
            if isinstance(self.created_at, str)
            else self.created_at
        )

    def __str__(self) -> str:
        return (
            f' {self.is_completed and ' ' or ' '} {self.content}'
            f'{self.due and f'  [  {self.due.string}]' or ''}'
            f'{self.labels and f'  [ {','.join(self.labels)}]' or ''}'
            f'{self.priority and f'  [ {self.priority.name}]' or ''}'
        )


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


A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


def compose2(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    def inner(x: A) -> C:
        return f(g(x))

    return inner


def get_object_url_path(url: str) -> str:
    return f'{BASE_URL}/{url.removeprefix(BASE_URL + '/').partition('/')[0]}'


def create_url(url: str, params: dict) -> str:
    return f'{url}?{urllib.parse.urlencode(params)}'


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                         Globals                          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


BASE_URL = "https://api.todoist.com/rest/v2"
TASK_FIELDS = [f.name for f in fields(Task)]
PROJECT_FIELDS = [f.name for f in fields(Project)]
DUE_FIELDS = [f.name for f in fields(Due)]

TODO_STATE = TodoState(
    get_path(XdgDirs.DATA) / 'todoist' / 'creds.json',
    get_path(XdgDirs.STATE) / 'todoist' / 'projects.json',
    get_path(XdgDirs.STATE) / 'todoist' / 'tasks.json',
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
    url: str,
    data: dict | None = None,
    method: str = 'GET',
    headers: dict[str, str] = {},
) -> HTTPResponse:
    default_headers = {
        'Content-Type': 'application/json',
        'X-Request-Id': f'{uuid.uuid4()}',
        'Authorization': f'Bearer {get_token()}',
    }
    req = urllib.request.Request(
        url,
        headers=default_headers | headers,
        data=data and json.dumps(data, default=str).encode('utf-8'),
        method=method,
    )
    try:
        res = urllib.request.urlopen(req, timeout=5)
        if not 200 <= res.status <= 300:
            raise Exception(
                f'Todoist responded with code: {res.status}, content: {res.read().decode('utf-8')}'
            )
        return res
    except urllib.error.URLError as e:
        bail(f'ERROR: Request failed {url}, REASON: {e}', ExitCode.REQUEST_FAILED)


def download(file: Path, res: HTTPResponse) -> None:
    with open(file, 'wb') as f:
        # copy 16 KB chunk
        chunk_size = 1024 * 16
        with res as reader:
            while chunk := reader.read(chunk_size):
                f.write(chunk)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                        Adapters                          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def dict2Project(d: dict) -> Project:
    return Project(**{f: v for f, v in d.items() if f in PROJECT_FIELDS})


def dict2Task(d: dict) -> Task:
    task_fields = {f: v for f, v in d.items() if f in TASK_FIELDS}
    task_fields['due'] = {
        f.removeprefix('due_'): v for f, v in d.items() if f.startswith('due_')
    } or task_fields.get('due')
    return Task(**(task_fields))


def task2Dict(t: Task) -> dict:
    task_dict = asdict(t)
    task_dict.pop('id', None)
    due_dict = task_dict.pop('due', {}) or {}
    return task_dict | {'due_' + k: v for k, v in due_dict.items()}


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    Object Operations                     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def download_objects(url: str, state_file: Path, exit_code: ExitCode) -> None:
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
        download_objects(url, state_file, exit_code)
    try:
        with open(state_file) as f:
            return json.load(f)
    except Exception as e:
        bail(f'ERROR: Failed to read {state_file.name}, REASON: {e}', exit_code)


def update_object(
    url: str,
    data: dict | None,
    state_file: Path,
    exit_code: ExitCode,
    method: str = 'POST',
) -> None:
    request(url, data, method)
    download_objects(get_object_url_path(url), state_file, exit_code)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                         Objects                          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def get_projects() -> list[Project]:
    return [
        dict2Project(p)
        for p in get_object(
            f'{BASE_URL}/projects',
            TODO_STATE.projects,
            ExitCode.PROJECT_NOT_FOUND,
        )
    ]


def get_active_tasks() -> list[Task]:
    return [
        dict2Task(t)
        for t in get_object(
            f'{BASE_URL}/tasks',
            TODO_STATE.tasks,
            ExitCode.TASK_NOT_FOUND,
        )
    ]


custom_sort_tasks = compose2(
    partial(enumerate, start=1),
    partial(sorted, key=lambda t: t.created_at, reverse=True),
)


def update_task(task: Task) -> None:
    update_object(
        f"{BASE_URL}/tasks/{task.id}" if task.id else f"{BASE_URL}/tasks",
        task2Dict(task),
        TODO_STATE.tasks,
        ExitCode.TASK_NOT_FOUND,
    )


update_task_from_dict = compose2(update_task, dict2Task)


def delete_task(task: Task) -> None:
    update_object(
        f"{BASE_URL}/tasks/{task.id}",
        None,
        TODO_STATE.tasks,
        ExitCode.TASK_NOT_FOUND,
        'DELETE',
    )


def close_task(task: Task) -> None:
    print(f"{BASE_URL}/tasks/{task.id}/close")
    update_object(
        f"{BASE_URL}/tasks/{task.id}/close",
        None,
        TODO_STATE.tasks,
        ExitCode.TASK_NOT_FOUND,
    )


def display_projects() -> None:
    for p in get_projects():
        print(p)


def display_tasks() -> None:
    for i, t in custom_sort_tasks(get_active_tasks()):
        print(f'{i:>4}. {t}')


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                       Controllers                        ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def get_project_controller(args: argparse.Namespace) -> None:
    display_projects()


def get_task_controller(args: argparse.Namespace) -> None:
    url = f'{BASE_URL}/tasks'
    params = {}
    if args.filter:
        params['filter'] = args.filter
    if args.label:
        params['label'] = args.label
    if args.project:
        params['project_id'] = [p.id for p in get_projects() if p.name == args.project][
            0
        ]
    if params:
        download_objects(
            create_url(url, params),
            TODO_STATE.tasks,
            ExitCode.TASK_NOT_FOUND,
        )
    display_tasks()


def add_task_controller(args: argparse.Namespace) -> None:
    # TODO: validate section
    task_fields = vars(args)
    task_fields['project_id'] = [
        p.id for p in get_projects() if p.name == args.project
    ][0]
    update_task_from_dict(task_fields)


def delete_task_controller(args: argparse.Namespace) -> None:
    tasks = get_active_tasks()
    tasks_to_delete = [t for i, t in custom_sort_tasks(tasks) if i == args.index]
    if not tasks_to_delete:
        bail(f'ERROR: Index not in range 1..{len(tasks)}', ExitCode.TASK_NOT_FOUND)
    delete_task(tasks_to_delete[0])


def update_task_controller(args: argparse.Namespace) -> None:
    tasks = get_active_tasks()
    tasks_to_update: list[Task] = [
        t for i, t in custom_sort_tasks(tasks) if i == args.index
    ]
    if not tasks_to_update:
        bail(f'ERROR: Index not in range 1..{len(tasks)}', ExitCode.TASK_NOT_FOUND)

    update_task_from_dict(
        asdict(tasks_to_update[0]) | {k: v for k, v in vars(args).items() if v}
    )


def close_task_controller(args: argparse.Namespace) -> None:
    tasks = get_active_tasks()
    tasks_to_close: list[Task] = [
        t for i, t in custom_sort_tasks(tasks) if i == args.index
    ]
    if not tasks_to_close:
        bail(f'ERROR: Index not in range 1..{len(tasks)}', ExitCode.TASK_NOT_FOUND)

    close_task(tasks_to_close[0])

def list_controller(args: argparse.Namespace) -> None:
    if args.project:
        for p in get_projects():
            print(p.name)
    elif args.labels:
        for l in Label:
            print(l.value)
    elif args.priority:
        for l in Priority:
            print(l.value)
    elif args.task:
        for t in get_active_tasks():
            print(t.content)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                  Command Line Options                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def main(argv: Sequence[str] | None = None) -> int:
    # valid values
    valid_projects = [p.name for p in get_projects()]
    valid_labels = [l.value for l in Label]
    valid_priorities = [p.value for p in Priority]
    tasks_length = len(get_active_tasks())

    parser = argparse.ArgumentParser(
        description='A client to interact with Todoist', epilog='Happy todoing'
    )
    subparser = parser.add_subparsers(required=True)

    # -- get sub-command
    get_parser = subparser.add_parser('get', help='Get a Todoist object')
    get_obj_parser = get_parser.add_subparsers(required=True)
    # -- get project sub-command
    get_project_parser = get_obj_parser.add_parser('project', help='Get a project')
    get_project_parser.add_argument(
        '-a', '--all', action='store_true', help='Get all projects'
    )
    get_project_parser.set_defaults(func=get_project_controller)

    # -- get task sub-command
    get_task_parser = get_obj_parser.add_parser('task', help='Get a task')
    get_task_parser.add_argument(
        '-f', '--filter', help='Filter task by an supported filter'
    )
    get_task_parser.add_argument(
        '-l', '--label', help='Filter task by label', choices=valid_labels
    )
    get_task_parser.add_argument(
        '-p', '--project', help='Filter task by project', choices=valid_projects
    )
    get_task_parser.set_defaults(func=get_task_controller)

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
        choices=valid_projects,
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
        choices=valid_labels,
    )
    add_task_parser.add_argument(
        '-p',
        '--priority',
        type=int,
        help='Add task with priority',
        default=Priority.P4,
        choices=valid_priorities,
    )
    add_task_date_group = add_task_parser.add_mutually_exclusive_group()
    add_task_date_group.add_argument(
        '-d',
        '--due-string',
        help='Task due by',
    )
    add_task_date_group.add_argument(
        '--due-date',
        type=date.fromisoformat,
        help='Task due by date',
    )
    add_task_date_group.add_argument(
        '--due-datetime',
        type=datetime.fromisoformat,
        help='Task due by datetime',
    )
    add_task_parser.set_defaults(func=add_task_controller)

    # -- delete sub-command
    delete_parser = subparser.add_parser('delete', help='Delete a Todoist object')
    delete_obj_parser = delete_parser.add_subparsers(required=True)

    delete_task_parser = delete_obj_parser.add_parser('task', help='Delete a task')
    delete_task_parser.add_argument(
        'index',
        type=int,
        help=f'Delete task by index in the range 1..{tasks_length}',
    )
    delete_task_parser.set_defaults(func=delete_task_controller)

    # -- update sub-command
    update_parser = subparser.add_parser('update', help='Update a Todoist object')
    update_obj_parser = update_parser.add_subparsers(required=True)

    update_task_parser = update_obj_parser.add_parser('task', help='Update a task')
    update_task_parser.add_argument(
        'index',
        type=int,
        help=f'Update task by index in the range 1..{tasks_length}',
    )

    update_task_parser.add_argument(
        '-c',
        '--content',
        help='Update task content',
    )
    update_task_parser.add_argument(
        '--description',
        help='Update task description',
    )
    update_task_parser.add_argument(
        '-l',
        '--labels',
        type=Label,
        action='append',
        default=[],
        help='Update task labels',
    )
    update_task_parser.add_argument(
        '-p',
        '--priority',
        type=int,
        help='Update task Priority',
    )
    update_task_date_group = update_task_parser.add_mutually_exclusive_group()
    update_task_date_group.add_argument(
        '-d',
        '--due-string',
        help='Update task due',
    )
    update_task_date_group.add_argument(
        '--due-date',
        type=date.fromisoformat,
        help='Update task due date',
    )
    update_task_date_group.add_argument(
        '--due-datetime',
        type=datetime.fromisoformat,
        help='Update task due datetime',
    )
    update_task_parser.set_defaults(func=update_task_controller)

    # -- close sub-command
    close_parser = subparser.add_parser('close', help='Close a Todoist object')
    close_obj_parser = close_parser.add_subparsers(required=True)

    close_task_parser = close_obj_parser.add_parser('task', help='Close a task')
    close_task_parser.add_argument(
        'index',
        type=int,
        help=f'close task by index in the range 1..{tasks_length}',
    )
    close_task_parser.set_defaults(func=close_task_controller)

    # -- list sub-command
    list_parser = subparser.add_parser('list', help='list a Todoist object')
    list_group_parser = list_parser.add_mutually_exclusive_group(required=True)
    list_group_parser.add_argument('-p', '--project', action='store_true', help='List projects')
    list_group_parser.add_argument('-t', '--task', action='store_true', help='List tasks')
    list_group_parser.add_argument('-l', '--labels', action='store_true', help='List labels')
    list_group_parser.add_argument('--priority', action='store_true', help='List priority')
    list_group_parser.set_defaults(func=list_controller)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
