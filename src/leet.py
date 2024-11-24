#!/usr/bin/env python3
"""This program sets up everything for the daily leetcode problems"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
import sys
import urllib.error
import urllib.request
import webbrowser
from contextlib import closing
from pathlib import Path
from typing import Sequence

TODAY = datetime.datetime.now()
LEET_DAILY_DIR = (
    Path.home()
    / "playground"
    / "projects"
    / "learn"
    / "competitive_programming"
    / f"{TODAY:%Y}"
    / f"{TODAY:%B}".lower()
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="leet",
        description="leet helps with doing leetcode daily",
        epilog="Happy leetcoding",
    )
    parser.add_argument(
        "-b", "--browser", action="store_false", help="do not open browser"
    )
    parser.add_argument(
        "-f", "--file", action="store_false", help="do not create a file"
    )
    parser.add_argument(
        "-n", "--neovim", action="store_false", help="do not open neovim"
    )
    args = parser.parse_args(argv)

    try:
        daily_qn_link = get_daily_qn_link()
    except urllib.error.URLError:
        print(
            f'Unable to get response from leetcode. Check your network connection',
            file=sys.stderr,
        )
        return 1

    leet_file = LEET_DAILY_DIR / Path(daily_qn_link).with_suffix(".py").name

    if args.browser:
        open_browser(daily_qn_link)
    if args.file:
        create_file(leet_file, daily_qn_link)
    if args.neovim:
        subprocess.run(["nvim", str(leet_file)])

    return 0


def open_browser(daily_qn_link: str) -> None:
    # if running WSL launch windows chrome
    if platform.release().endswith('microsoft-standard-WSL2'):
        chrome_executable = Path(
            '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe'
        )
        if not chrome_executable.exists():
            return
        subprocess.run([str(chrome_executable), daily_qn_link])
        return
    webbrowser.open(daily_qn_link)


def get_daily_qn_link() -> str:
    base_url = "https://leetcode.com/graphql/"
    query = {
        "query": "query questionOfToday {\n\tactiveDailyCodingChallengeQuestion {\n\t\tdate\n\t\tlink\n\t}\n}\n",
        "operationName": "questionOfToday",
    }
    # query_enc = urllib.parse.urlencode(query).encode('utf-8')
    req = urllib.request.Request(base_url, json.dumps(query).encode("utf-8"))
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0")
    req.add_header("Accept", "*/*")
    with closing(urllib.request.urlopen(req, timeout=5.0)) as res:
        res_data = json.loads(res.read())
        daily_qn = res_data["data"]["activeDailyCodingChallengeQuestion"]["link"]
        return base_url.removesuffix("/graphql/") + daily_qn


def create_file(leet_file: Path, daily_qn_link: str):
    if leet_file.exists():
        print(f'File already exits {leet_file}')
        return leet_file
    leet_file.parent.mkdir(parents=True, exist_ok=True)
    with open(leet_file, "w") as f:
        f.write(
            f'''\
"""
Created Date: {TODAY:%Y-%m-%d}
Qn: 
Link: {daily_qn_link}
Notes:
"""

import unittest


class Solution:
    def main(self):
        pass


class TestSolution(unittest.TestCase):
    def setUp(self):
        self.sol = Solution()

    def test_main(self):
        pass


if __name__ == '__main__':
    unittest.main()
'''
        )
        print(f'File created {leet_file}')


if __name__ == "__main__":
    raise SystemExit(main())
