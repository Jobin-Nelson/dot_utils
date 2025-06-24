#!/usr/bin/env python3
#
# ██████╗ ██████╗ ██╗    ██╗
# ██╔══██╗██╔══██╗██║    ██║
# ██████╔╝██████╔╝██║ █╗ ██║
# ██╔══██╗██╔══██╗██║███╗██║
# ██████╔╝██████╔╝╚███╔███╔╝
# ╚═════╝ ╚═════╝  ╚══╝╚══╝
#

'''
Script to backup before wipe
'''

from __future__ import annotations

import asyncio
import itertools
import shutil
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    Global Variables                      ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Actual workers are NUM_WORKERS * 4 (add, commit, push, status, result)
NUM_WORKERS = 5
QUEUE_MAX_SIZE = 25


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                        VCS Types                         ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


@dataclass
class Git:
    cwd: Path
    git_dir: Path | None = None
    work_tree: Path | None = None
    cmd: list[str] = field(init=False)
    proc: asyncio.subprocess.Process | None = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self):
        self.cmd = ['git', '-C', str(self.cwd)]
        if self.git_dir:
            self.cmd.extend(['--git-dir', str(self.git_dir)])
        if self.work_tree:
            self.cmd.extend(['--work-tree', str(self.work_tree)])

    async def add(self):
        add_args = ['add', '-u']
        self.proc = await exec_cmd(self.cmd + add_args)

    async def commit(self):
        commit_args = [
            'commit',
            '--no-gpg-sign',
            '-a',
            '-m',
            'chore: bbw.py backup commit',
        ]
        self.proc = await exec_cmd(self.cmd + commit_args)

    async def push(self):
        push_args = ['push', 'origin', 'HEAD']
        self.proc = await exec_cmd(self.cmd + push_args)

    async def status(self):
        status_args = ['status', '--porcelain']
        if self.git_dir is None:
            status_args.append('--untracked-files=normal')
        cmd = self.cmd + status_args
        self.proc = await exec_cmd(cmd, True)


async def exec_cmd(cmd: list[str], cap_output: bool = False):
    pipe = asyncio.subprocess.PIPE if cap_output else asyncio.subprocess.DEVNULL
    return await asyncio.create_subprocess_exec(
        *cmd,
        stdout=pipe,
        stderr=asyncio.subprocess.DEVNULL,
    )


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                         Workers                          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


async def _producer(
    batch: list[Git], work_queue: asyncio.Queue, producer_completed: asyncio.Event
):
    for job in batch:
        await work_queue.put(job)
    producer_completed.set()


async def _worker(
    work_queue: asyncio.Queue, result_queue: asyncio.Queue, callback: Callable
):
    while True:
        task = await work_queue.get()
        if task.proc is not None:
            await task.proc.wait()
        await callback(task)()
        await result_queue.put(task)
        work_queue.task_done()


async def _janitor(queue: asyncio.Queue):
    while True:
        task = await queue.get()
        stdout, _ = await task.proc.communicate()
        if stdout:
            print(f'{task.cwd} repo is dirty')
        queue.task_done()


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                        Callbacks                         ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def add_callback(git: Git):
    return git.add


def commit_callback(git: Git):
    return git.commit


def push_callback(git: Git):
    return git.push


def status_callback(git: Git):
    return git.status


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                     Worker Handlers                      ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def hook_workers_handler(
    tasks: list, in_queue: asyncio.Queue, out_queue: asyncio.Queue, callback: Callable
):
    for _ in range(NUM_WORKERS):
        tasks.append(asyncio.create_task(_worker(in_queue, out_queue, callback)))


def hook_cleanup_handler(tasks: list, end_queue: asyncio.Queue):
    for _ in range(NUM_WORKERS):
        tasks.append(asyncio.create_task(_janitor(end_queue)))


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                     Core Functions                       ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


async def _controller() -> int:
    home = Path.home()
    dotfiles = home / '.dotfiles'
    project_path = home / 'playground' / 'projects'
    extra_paths = [
        home / '.config' / 'nvim',
        home / '.password-store',
        home / '.config' / 'nixos-config',
        home / 'playground' / 'lab',
    ]

    projects = [p for p in project_path.iterdir() if p.is_dir()]
    extra_repos = [p for p in extra_paths if p.is_dir()]

    batch = [Git(p) for p in itertools.chain(projects, extra_repos)] + [
        Git(home, dotfiles, home)
    ]

    tasks = []
    add_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    commit_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    push_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    status_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    result_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)

    producer_completed = asyncio.Event()
    producer_completed.clear()

    await _producer(batch, add_queue, producer_completed)

    hook_workers = partial(hook_workers_handler, tasks)
    hook_workers(add_queue, commit_queue, add_callback)
    hook_workers(commit_queue, push_queue, commit_callback)
    hook_workers(push_queue, status_queue, push_callback)
    hook_workers(status_queue, result_queue, status_callback)
    hook_cleanup_handler(tasks, result_queue)

    await producer_completed.wait()
    await add_queue.join()
    await commit_queue.join()
    await push_queue.join()
    await status_queue.join()
    await result_queue.join()

    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)

    return 0


def main() -> int:
    return asyncio.run(_controller())


def check_requirements():
    executables = [
        'git',
        # 'gclone.sh',
    ]
    not_found = [e for e in executables if not shutil.which(e)]
    if not_found:
        raise SystemExit(f"Executable {', '.join(not_found)} not found in PATH")


if __name__ == '__main__':
    check_requirements()
    raise SystemExit(main())
