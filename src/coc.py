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
class Boost:
    value: int
    duration: timedelta = timedelta(minutes=BOOST_DURATION)

    def __add__(self, other: Boost) -> Boost:
        if self.value != other.value:
            bail(
                'ERROR: Cannot add boosts with different values',
                ExitCode.INCOMPATIBLE_BOOST,
            )
        return Boost(self.value, self.duration + other.duration)


@dataclass
class Upgrade:
    time_left: timedelta

    def max_boost_time(self, boost: Boost) -> Upgrade:
        task_minutes = timedelta2min(self.time_left)

        return Upgrade(timedelta(minutes=task_minutes // boost.value))

    def boosts_required(self, boost: Boost) -> int:
        task_minutes = timedelta2min(self.time_left)
        boost_minutes = timedelta2min(boost.duration)
        boosted_minutes = boost_minutes * boost.value
        n, res = divmod(task_minutes, boosted_minutes)
        return n + (1 if res else 0)

    def apply_boost(self, boost: Boost) -> Upgrade:
        task_minutes = timedelta2min(self.time_left)
        boost_minutes = timedelta2min(boost.duration)

        boosted_minutes = boost_minutes * boost.value

        return (
            Upgrade(timedelta(minutes=task_minutes - boosted_minutes + boost_minutes))
            if task_minutes > boosted_minutes
            else self.max_boost_time(boost)
        )


class ExitCode(IntEnum):
    PARSE_FAILED = auto()
    INCOMPATIBLE_BOOST = auto()


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    Utility Functions                     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def bail(message: str, code: ExitCode):
    print(message, file=sys.stderr)
    raise SystemExit(code.value)


def get_boost_value(upgrade_type: UpgradeType) -> int:
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
    boost_value = get_boost_value(upgrade_type)

    now = datetime.now()

    upgrade = Upgrade(args.time_left)
    rem_boost = Boost(boost_value, args.boost)
    boost = Boost(boost_value)
    rem_boost_upgrade = upgrade.apply_boost(rem_boost)

    rem_boost_time_left = rem_boost_upgrade.time_left
    boosted_time_left = upgrade.apply_boost(boost + rem_boost).time_left
    max_boosted_time_left = upgrade.max_boost_time(boost).time_left
    max_boosts_required = rem_boost_upgrade.boosts_required(boost)

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
        upgrade = Upgrade(time_left)
        boost = Boost(BUILD_BOOST)
        expected = timedelta(hours=1)

        self.assertEqual(expected, upgrade.apply_boost(boost).time_left)

    def test_lab_boost(self):
        time_left = timedelta(hours=24)
        upgrade = Upgrade(time_left)
        boost = Boost(LAB_BOOST)
        expected = timedelta(hours=1)

        self.assertEqual(expected, upgrade.apply_boost(boost).time_left)

    def test_build_boost_with_minutes(self):
        time_left = timedelta(hours=12, minutes=23)
        upgrade = Upgrade(time_left)
        boost = Boost(BUILD_BOOST)
        expected = timedelta(hours=3, minutes=23)

        self.assertEqual(expected, upgrade.apply_boost(boost).time_left)

    def test_lab_boost_with_minutes(self):
        time_left = timedelta(days=1, hours=20, minutes=23)
        upgrade = Upgrade(time_left)
        boost = Boost(LAB_BOOST)
        expected = timedelta(hours=21, minutes=23)

        self.assertEqual(expected, upgrade.apply_boost(boost).time_left)

    def test_build_boost_with_boost_left(self):
        time_left = timedelta(hours=15, minutes=23)
        upgrade = Upgrade(time_left)
        boost = Boost(BUILD_BOOST) + Boost(BUILD_BOOST, timedelta(minutes=30))
        expected = timedelta(hours=1, minutes=53)

        self.assertEqual(expected, upgrade.apply_boost(boost).time_left)

    def test_build_boost_with_no_boost_left(self):
        time_left = timedelta(hours=5)
        upgrade = Upgrade(time_left)
        boost = Boost(BUILD_BOOST) + Boost(BUILD_BOOST, timedelta(minutes=30))
        expected = timedelta(minutes=30)

        self.assertEqual(expected, upgrade.apply_boost(boost).time_left)


if __name__ == '__main__':
    raise SystemExit(main())
