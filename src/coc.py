#!/usr/bin/env python3
"""
Script to calculate boost time.
All units are in minutes
"""

import argparse
import re
import sys
import unittest
from dataclasses import dataclass
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


@dataclass
class Upgrade:
    boost: int
    time_left: timedelta
    rem_boost: timedelta = timedelta(0)

    def boost_time(self, num_boost: int = 1) -> timedelta:
        task_minutes = timedelta2min(self.time_left)
        rem_boost_minutes = timedelta2min(self.rem_boost)

        boostable_minutes = BOOST_DURATION * num_boost
        total_boost_minutes = boostable_minutes + rem_boost_minutes
        total_boosted_minutes = total_boost_minutes * self.boost

        return (
            timedelta(
                minutes=task_minutes - total_boosted_minutes + total_boost_minutes
            )
            if task_minutes > total_boosted_minutes
            else self.max_boost_time()
        )

    def max_boost_time(self) -> timedelta:
        task_minutes = timedelta2min(self.time_left)

        return timedelta(minutes=task_minutes // self.boost)

    def boosts_required(self) -> int:
        task_minutes = timedelta2min(self.time_left)
        rem_boost_minutes = timedelta2min(self.rem_boost)
        covered_by_rem = rem_boost_minutes * self.boost

        rem_task_minutes = task_minutes - covered_by_rem

        if rem_task_minutes <= 0:
            return 0

        boosted_minutes = BOOST_DURATION * self.boost
        n, res = divmod(rem_task_minutes, boosted_minutes)
        return n + (1 if res else 0)


class ExitCode(IntEnum):
    PARSE_FAILED = auto()
    INCOMPATIBLE_BOOST = auto()


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    Utility Functions                     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def bail(message: str, code: ExitCode):
    print(message, file=sys.stderr)
    raise SystemExit(code.value)


def get_boost(upgrade_type: UpgradeType) -> int:
    match upgrade_type:
        case UpgradeType.BUILD:
            return BUILD_BOOST
        case UpgradeType.LAB:
            return LAB_BOOST


def timedelta2min(t: timedelta) -> int:
    return int(t.total_seconds()) // 60


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


def upgrade_times(upgrade_type: UpgradeType, args: argparse.Namespace) -> None:
    boost = get_boost(upgrade_type)

    now = datetime.now()

    time_left = args.time_left

    upgrade = Upgrade(boost, time_left, args.boost)

    rem_boost_time_left = upgrade.boost_time(0)
    boosted_time_left = upgrade.boost_time(args.number)
    max_boosted_time_left = upgrade.max_boost_time()
    max_boosts_required = upgrade.boosts_required()

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
        upgrade = Upgrade(BUILD_BOOST, time_left)
        expected = timedelta(hours=1)

        self.assertEqual(expected, upgrade.boost_time(1))

    def test_lab_boost(self):
        time_left = timedelta(hours=24)
        upgrade = Upgrade(LAB_BOOST, time_left)
        expected = timedelta(hours=1)

        self.assertEqual(expected, upgrade.boost_time(1))

    def test_build_boost_with_minutes(self):
        time_left = timedelta(hours=12, minutes=23)
        upgrade = Upgrade(BUILD_BOOST, time_left)
        expected = timedelta(hours=3, minutes=23)

        self.assertEqual(expected, upgrade.boost_time(1))

    def test_lab_boost_with_minutes(self):
        time_left = timedelta(days=1, hours=20, minutes=23)
        upgrade = Upgrade(LAB_BOOST, time_left)
        expected = timedelta(hours=21, minutes=23)

        self.assertEqual(expected, upgrade.boost_time(1))

    def test_build_boost_with_boost_left(self):
        time_left = timedelta(hours=15, minutes=23)
        boost_left = timedelta(minutes=30)
        upgrade = Upgrade(BUILD_BOOST, time_left, boost_left)
        expected = timedelta(hours=1, minutes=53)

        self.assertEqual(expected, upgrade.boost_time(1))

    def test_build_boost_with_no_boost_left(self):
        time_left = timedelta(hours=5)
        boost_left = timedelta(minutes=30)
        upgrade = Upgrade(BUILD_BOOST, time_left, boost_left)
        expected = timedelta(minutes=30)

        self.assertEqual(expected, upgrade.boost_time(1))


if __name__ == '__main__':
    raise SystemExit(main())
