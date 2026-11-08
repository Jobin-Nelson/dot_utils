#!/usr/bin/env python3
"""
Script to calculate boost time
"""

import argparse
from datetime import datetime, timedelta
from enum import StrEnum, auto
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

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                          Types                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


class UpgradeType(StrEnum):
    BUILD = auto()
    LAB = auto()


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                    Utility Functions                     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


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


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                   Core Implementation                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def boost_time(t: timedelta, boost: int, n: int = 1) -> timedelta:
    total_minutes = int(t.total_seconds()) // 60

    boosted_minutes = BOOST_DURATION * boost * n
    if total_minutes > boosted_minutes:
        return timedelta(minutes=total_minutes - boosted_minutes + BOOST_DURATION)
    else:
        return max_boost_time(t, boost)


def max_boost_time(t: timedelta, boost: int) -> timedelta:
    total_minutes = int(t.total_seconds()) // 60

    return timedelta(minutes=total_minutes // boost)


def boosts_required(t: timedelta, boost: int) -> int:
    total_minutes = int(t.total_seconds()) // 60
    boosted_minutes = BOOST_DURATION * boost
    return total_minutes // boosted_minutes + 1


def upgrade_times(upgrade_type: UpgradeType, args: argparse.Namespace) -> None:
    boost = 0
    match upgrade_type:
        case UpgradeType.BUILD:
            boost = BUILD_BOOST
        case UpgradeType.LAB:
            boost = LAB_BOOST

    now = datetime.now()

    time_left = timedelta(0)
    if args.days:
        time_left += args.days
    if args.hours:
        time_left += args.hours
    if args.minutes:
        time_left += args.minutes

    boosted_time_left = boost_time(time_left, boost, args.number)
    max_boosted_time_left = max_boost_time(time_left, boost)
    max_boosts_required = boosts_required(time_left, boost)

    # Time left without boost
    print(f'{dt2str(now + time_left)} |  0 boost | {td2str(time_left)}')
    # Time left with n boost
    print(
        f'{dt2str(now + boosted_time_left)} | {args.number:2} boost | {td2str(boosted_time_left)}'
    )
    # Time left with max boost
    print(
        f'{dt2str(now + max_boosted_time_left)} | {max_boosts_required:2} boost | {td2str(max_boosted_time_left)}'
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
        '-d',
        '--days',
        type=lambda x: timedelta(days=float(x)),
        help='Time left in days',
    )
    build.add_argument(
        '-H',
        '--hours',
        type=lambda x: timedelta(hours=float(x)),
        help='Time left in hours',
    )
    build.add_argument(
        '-m',
        '--minutes',
        type=lambda x: timedelta(minutes=float(x)),
        help='Time left in minutes',
    )
    build.add_argument(
        '-n',
        '--number',
        type=int,
        default=1,
        help='Number of boosts',
    )
    build.set_defaults(func=build_times)

    # Lab parser
    lab = subparser.add_parser(
        UpgradeType.LAB.value, aliases=[UpgradeType.LAB.value[0]]
    )
    lab.add_argument(
        '-d',
        '--days',
        type=lambda x: timedelta(days=float(x)),
        help='Time left in days',
    )
    lab.add_argument(
        '-H',
        '--hours',
        type=lambda x: timedelta(hours=float(x)),
        help='Time left in hours',
    )
    lab.add_argument(
        '-m',
        '--minutes',
        type=lambda x: timedelta(minutes=float(x)),
        help='Time left in minutes',
    )
    lab.add_argument(
        '-n',
        '--number',
        type=int,
        default=1,
        help='Number of boosts',
    )
    lab.set_defaults(func=lab_times)

    args = parser.parse_args(argv)

    args.func(args)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
