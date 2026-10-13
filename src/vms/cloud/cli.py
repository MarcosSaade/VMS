"""`python -m vms.cloud` (`scripts/gpu_session.sh`): GPU sessions on AWS.

    preflight                      checks that cost nothing: credentials, Region, the
                                   on-demand G vCPU quota, the AMI, a dry-run launch, and
                                   project resources left from earlier sessions
    run SCENARIO [SCENARIO ...]    a whole session: launch one tagged on-demand GPU instance,
                                   sync the code, run the scenarios there, copy the results
                                   back into bench/results/, terminate everything and verify by
                                   tag. Asks before launching (--yes skips the prompt).
    teardown --session ID          delete what a session left behind (from its tags and its
                                   saved state), then verify
    list                           project resources (tag project=vms) that still exist
    budget --email ADDRESS         create the monthly AWS Budgets alarm once (configs/cloud.yaml)

Exit codes: 0 ok; 1 setup or a scenario failed (teardown verified); 2 resources remain after
teardown (check now: they may be billing); 3 not confirmed, nothing created; 4 preflight failed;
130 interrupted (teardown verified).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from vms.cloud.aws import Boto3Cloud, BudgetSpec
from vms.cloud.remote import SshRemote, generate_key
from vms.cloud.session import GpuSession, Preflight, PreflightError
from vms.cloud.settings import SERVICE, CloudSettings
from vms.common.clock import SystemClock
from vms.common.ids import new_ulid
from vms.common.logs import configure_logging, get_logger
from vms.common.settings import load_settings

log = get_logger(__name__)

STATE_DIR = Path(".gpu-sessions")  # gitignored: per-session key, known_hosts, state
CHECKIP_URL = "https://checkip.amazonaws.com"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/gpu_session.sh",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="checks that cost nothing")
    run = commands.add_parser("run", help="a whole GPU session")
    run.add_argument("scenarios", nargs="+", help="bench/scenarios/*.yaml (e2e or model)")
    run.add_argument("--yes", action="store_true", help="do not ask before launching")
    run.add_argument("--allow-dirty", action="store_true", help="allow uncommitted changes")
    teardown = commands.add_parser("teardown", help="delete a session's leftovers")
    teardown.add_argument("--session", required=True)
    commands.add_parser("list", help="project resources that still exist")
    budget = commands.add_parser("budget", help="create the monthly Budgets alarm once")
    budget.add_argument("--email", required=True, help="where the alerts go")
    budget.add_argument("--amount", type=float, default=None, help="USD per month")
    return parser


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _ssh_cidr(settings: CloudSettings) -> str:
    if settings.ssh_cidr:
        return settings.ssh_cidr
    with urllib.request.urlopen(CHECKIP_URL, timeout=settings.timeouts.aws_connect_s) as reply:
        return f"{reply.read().decode().strip()}/32"


def _describe(pre: Preflight) -> str:
    lines = [
        f"  account / Region   {pre.identity.account} / {pre.identity.region}",
        f"  instance           {pre.instance_type} on-demand, {pre.vcpus} vCPUs "
        f"(quota {pre.vcpu_quota:g})",
        f"  AMI                {pre.image.id} ({pre.image.name})",
        f"  expected cost      ${pre.hourly_usd:.3f} per hour (instance + volume + IPv4), "
        "billed per second",
        f"  hard cap           the instance terminates itself after {pre.max_session_minutes} "
        f"min: at most ${pre.max_cost_usd:.2f}",
    ]
    if pre.existing:
        lines.append(f"  WARNING            {len(pre.existing)} project resources still exist:")
        lines += [f"                     {arn}" for arn in pre.existing]
    return "\n".join(lines)


def _confirm(pre: Preflight) -> bool:
    _out("GPU session about to launch:\n" + _describe(pre))
    try:
        answer = input("Type 'launch' to start it: ")
    except EOFError:
        return False
    return answer.strip() == "launch"


def _announce(pre: Preflight) -> bool:
    _out("GPU session launching (--yes):\n" + _describe(pre))
    return True


def _session(settings: CloudSettings, cloud: Boto3Cloud, session_id: str) -> GpuSession:
    t = settings.timeouts
    return GpuSession(
        settings,
        cloud,
        session_id=session_id,
        repo=Path.cwd(),
        state_dir=STATE_DIR,
        clock=SystemClock(),
        sleep=time.sleep,
        remote_factory=lambda host, key, known: SshRemote(
            host, settings.ssh_user, key, known, t.ssh_connect_s, t.poll_s
        ),
        keygen=generate_key,
        ssh_cidr=lambda: _ssh_cidr(settings),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = load_settings(CloudSettings, SERVICE)
    configure_logging(SERVICE, settings.log.level)
    cloud = Boto3Cloud(settings.region, settings.timeouts, os.environ.get("AWS_PROFILE"))

    if args.command == "list":
        found = [arn for arn in cloud.tagged_arns(settings.tags) if cloud.exists(arn)]
        _out("\n".join(found) if found else "no project resources exist")
        return 0

    if args.command == "budget":
        b = settings.budget
        spec = BudgetSpec(
            name=b.name,
            amount_usd=args.amount or b.amount_usd,
            email=args.email,
            actual_thresholds_pct=b.actual_thresholds_pct,
            forecast_thresholds_pct=b.forecast_thresholds_pct,
            tags=settings.tags,
        )
        account = cloud.identity().account
        if cloud.budget_exists(account, spec.name):
            _out(f"budget {spec.name!r} already exists; nothing changed")
            return 0
        cloud.create_budget(account, spec)
        _out(f"budget {spec.name!r} created: ${spec.amount_usd:.2f}/month, alerts to the address")
        return 0

    if args.command == "teardown":
        session = _session(settings, cloud, args.session)
        if session.state_path.is_file():
            session.state = type(session.state).load(session.state_path)
        session.adopt_tagged()
        report = session.teardown()
        _out("teardown verified" if report.verified else f"LEFTOVERS: {report.leftovers}")
        return 0 if report.verified else 2

    session = _session(settings, cloud, new_ulid())
    try:
        if args.command == "preflight":
            _out("preflight passed:\n" + _describe(session.preflight()))
            return 0
        _out(f"session {session.session_id}")
        code = session.run(
            args.scenarios,
            _announce if args.yes else _confirm,
            Path("bench/results"),
            allow_dirty=args.allow_dirty,
        )
    except PreflightError as exc:
        _out(f"preflight failed: {exc}")
        return 4
    if code == 2:
        _out(
            "\n!!! TEARDOWN NOT VERIFIED: resources tagged session="
            f"{session.session_id} still exist and may be billing.\n"
            f"!!! Run: scripts/gpu_session.sh teardown --session {session.session_id}\n"
        )
    return code
