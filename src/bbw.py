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
from pathlib import Path

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                        VCS Types                         ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


@dataclass
class Git:
    cwd: Path
    git_dir: Path | None = None
    work_tree: Path | None = None
    cmd: list[str] = field(init=False)

    def __post_init__(self):
        self.cmd = ['git', '-C', str(self.cwd)]
        if self.git_dir:
            self.cmd.extend(['--git-dir', str(self.git_dir)])
        if self.work_tree:
            self.cmd.extend(['--work-tree', str(self.work_tree)])

    async def commit(self):
        commit_args = [
            'commit',
            '--no-gpg-sign',
            '-a',
            '-m',
            'chore: bbw.py backup commit',
        ]
        await exec_cmd(self.cmd + commit_args)

    async def push(self):
        push_args = ['push', 'origin', 'HEAD']
        await exec_cmd(self.cmd + push_args)

    async def status(self):
        status_args = ['status', '--porcelain']
        if self.git_dir is None:
            status_args.append('--untracked-files=normal')
        cmd = self.cmd + status_args
        stdout = await exec_cmd(cmd, True)
        if stdout:
            print(f'{self.cwd} repo is dirty')


async def exec_cmd(cmd: list[str], cap_output: bool = False) -> bytes:
    pipe = asyncio.subprocess.PIPE if cap_output else asyncio.subprocess.DEVNULL
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=pipe,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    return stdout


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                         Workers                          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


async def worker(git: Git):
    await git.commit()
    await git.push()
    await git.status()


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                     Core Functions                       ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def get_repos() -> list[Git]:
    home = Path.home()
    dotfiles = home / '.dotfiles'
    project_path = home / 'playground' / 'projects'
    extra_repos = [
        home / '.config' / 'nvim',
        home / '.password-store',
        home / '.config' / 'nixos-config',
        home / 'playground' / 'lab',
    ]

    projects = [p for p in project_path.iterdir() if p.is_dir()]
    extra_repos = [p for p in extra_repos if p.is_dir()]

    return [Git(p) for p in itertools.chain(projects, extra_repos)] + [
        Git(home, dotfiles, home)
    ]


async def controller():
    repos = get_repos()

    async with asyncio.TaskGroup() as tg:
        for repo in repos:
            tg.create_task(worker(repo))


def main() -> int:
    asyncio.run(controller())
    return 0


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
