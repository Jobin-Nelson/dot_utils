#!/usr/bin/env python3
"""
Script to calculate boost time
"""

import argparse
import re
import sys
import unittest
from datetime import datetime, timedelta
from enum import IntEnum, StrEnum, auto
from functools import partial
from typing import Sequence

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    Global Variables                      ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Boost per hour
BUILD_BOOST = 10
LAB_BOOST = 24
BOOST_DURATION = 60

# TIME FORMAT
TIME_FORMAT = '%a %Y-%m-%d %I:%M %p'

# Match groups for days, hours, minutes
TIMEDELTA_REGEX = re.compile(
    r'(?:(?P<days>\d+)d)?'
    r'(?:(?P<hours>\d+)h)?'
    r'(?:(?P<minutes>\d+)m)?'
)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                          Types                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


class UpgradeType(StrEnum):
    BUILD = auto()
    LAB = auto()


class ExitCode(IntEnum):
    PARSE_FAILED = auto()


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    Utility Functions                     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def bail(message: str, code: ExitCode):
    print(message, file=sys.stderr)
    raise SystemExit(code.value)


def dt2str(t: datetime) -> str:
    return t.strftime(TIME_FORMAT)


def td2str(t: timedelta) -> str:
    days = t.days
    total_seconds = int(t.total_seconds())

    hours = (total_seconds // 3600) % 24
    minutes = (total_seconds // 60) % 60

    parts = []
    if days > 0:
        parts.append(f'{days}d')
    if hours > 0:
        parts.append(f'{hours}h')
    if minutes > 0:
        parts.append(f'{minutes}m')
    return ', '.join(parts) if parts else '0 minutes'


def parse_timedelta(time_str: str) -> timedelta | None:
    parts = TIMEDELTA_REGEX.match(time_str.strip().lower())
    if not parts:
        bail('ERROR: Unable to parse time', ExitCode.PARSE_FAILED)

    time_params = {k: int(v) for k, v in parts.groupdict().items() if v}  # pyright: ignore[reportOptionalMemberAccess]
    return timedelta(**time_params)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                   Core Implementation                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def boost_time(
    t: timedelta, boost: int, num_boosts: int = 1, rem_boost: timedelta = timedelta(0)
) -> timedelta:
    total_minutes = int(t.total_seconds()) // 60
    rem_boost_minutes = int(rem_boost.total_seconds()) // 60

    boostable_minutes = BOOST_DURATION * num_boosts
    total_boost_minutes = boostable_minutes + rem_boost_minutes
    total_boosted_minutes = total_boost_minutes * boost

    return (
        timedelta(minutes=total_minutes - total_boosted_minutes + total_boost_minutes)
        if total_minutes > total_boosted_minutes
        else max_boost_time(t, boost)
    )


def max_boost_time(t: timedelta, boost: int) -> timedelta:
    total_minutes = int(t.total_seconds()) // 60

    return timedelta(minutes=total_minutes // boost)


def boosts_required(t: timedelta, boost: int, rem_boost: timedelta) -> int:
    task_minutes = int(t.total_seconds()) // 60
    rem_boost_minutes = int(rem_boost.total_seconds()) // 60
    covered_by_rem = rem_boost_minutes * boost

    rem_task_minutes = task_minutes - covered_by_rem

    if rem_task_minutes <= 0:
        return 0

    boosted_minutes = BOOST_DURATION * boost
    n, res = divmod(rem_task_minutes, boosted_minutes)
    return n + (1 if res else 0)


def upgrade_times(upgrade_type: UpgradeType, args: argparse.Namespace) -> None:
    boost = 0
    match upgrade_type:
        case UpgradeType.BUILD:
            boost = BUILD_BOOST
        case UpgradeType.LAB:
            boost = LAB_BOOST

    now = datetime.now()

    time_left = args.time_left

    rem_boost_time_left = boost_time(time_left, boost, 0, args.boost)
    boosted_time_left = boost_time(time_left, boost, args.number, args.boost)
    max_boosted_time_left = max_boost_time(time_left, boost)
    max_boosts_required = boosts_required(time_left, boost, args.boost)

    # Headers
    print('      DATETIME          | BOOST | REM TIME')
    print('      --------          | ----- | --------')

    # Time left without boost
    print(
        f'{dt2str(now + rem_boost_time_left)} |   0   | {td2str(rem_boost_time_left)}'
    )
    # Time left with n boost
    print(
        f'{dt2str(now + boosted_time_left)} |  {args.number:2}   | {td2str(boosted_time_left)}'
    )
    # Time left with max boost
    print(
        f'{dt2str(now + max_boosted_time_left)} |  {max_boosts_required:2}   | {td2str(max_boosted_time_left)}'
    )


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                         Parser                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='COC', description='Helper for calculating boost timing'
    )
    subparser = parser.add_subparsers(required=True)

    build_times = partial(upgrade_times, UpgradeType.BUILD)
    lab_times = partial(upgrade_times, UpgradeType.LAB)

    # Build parser
    build = subparser.add_parser(
        UpgradeType.BUILD.value, aliases=[UpgradeType.BUILD.value[0]]
    )
    build.add_argument(
        'time_left',
        type=parse_timedelta,
        help="Time left in dhm format",
    )
    build.add_argument(
        '-n',
        '--number',
        type=int,
        default=1,
        help='Number of boosts',
    )
    build.add_argument(
        '-b',
        '--boost',
        type=parse_timedelta,
        default=timedelta(0),
        help='boost left',
    )
    build.set_defaults(func=build_times)

    # Lab parser
    lab = subparser.add_parser(
        UpgradeType.LAB.value, aliases=[UpgradeType.LAB.value[0]]
    )
    lab.add_argument(
        'time_left',
        type=parse_timedelta,
        help="Time left in dhm format",
    )
    lab.add_argument(
        '-n',
        '--number',
        type=int,
        default=1,
        help='Number of boosts',
    )
    lab.add_argument(
        '-b',
        '--boost',
        type=parse_timedelta,
        default=timedelta(0),
        help='boost left',
    )
    lab.set_defaults(func=lab_times)

    args = parser.parse_args(argv)

    args.func(args)

    return 0


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                       Unit Tests                         ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


class TestTimeParse(unittest.TestCase):
    def test_dhm(self):
        t = '2d3h5m'
        expected = timedelta(days=2, hours=3, minutes=5)
        self.assertEqual(expected, parse_timedelta(t))

    def test_hm_with_multiple_digits(self):
        t = '3h10m'
        expected = timedelta(hours=3, minutes=10)
        self.assertEqual(expected, parse_timedelta(t))


class TestBoost(unittest.TestCase):
    def test_build_boost(self):
        time_left = timedelta(hours=10)
        expected = timedelta(hours=1)

        self.assertEqual(expected, boost_time(time_left, BUILD_BOOST))

    def test_lab_boost(self):
        time_left = timedelta(hours=24)
        expected = timedelta(hours=1)

        self.assertEqual(expected, boost_time(time_left, LAB_BOOST))

    def test_build_boost_with_minutes(self):
        time_left = timedelta(hours=12, minutes=23)
        expected = timedelta(hours=3, minutes=23)

        self.assertEqual(expected, boost_time(time_left, BUILD_BOOST))

    def test_lab_boost_with_minutes(self):
        time_left = timedelta(days=1, hours=20, minutes=23)
        expected = timedelta(hours=21, minutes=23)

        self.assertEqual(expected, boost_time(time_left, LAB_BOOST))

    def test_build_boost_with_boost_left(self):
        time_left = timedelta(hours=15, minutes=23)
        boost_left = timedelta(minutes=30)
        expected = timedelta(hours=1, minutes=53)

        self.assertEqual(expected, boost_time(time_left, BUILD_BOOST, 1, boost_left))

    def test_build_boost_with_no_boost_left(self):
        time_left = timedelta(hours=5)
        boost_left = timedelta(minutes=30)
        expected = timedelta(minutes=30)

        self.assertEqual(expected, boost_time(time_left, BUILD_BOOST, 1, boost_left))


if __name__ == '__main__':
    raise SystemExit(main())
