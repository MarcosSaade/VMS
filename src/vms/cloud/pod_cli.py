"""`python -m vms.cloud.pod_cli` (`scripts/runpod_session.sh`): GPU sessions on Runpod pods (D37).

    preflight                      checks that cost nothing: the API key, the GPU's catalog price
                                   and stock, and project pods left from earlier sessions
    run SCENARIO [SCENARIO ...]    a whole session: create one pod, sync the code, run the
                                   model and ladder scenarios there, copy the results back into
                                   bench/results/, delete the pod and verify. Asks before
                                   creating it (--yes skips the prompt).
    teardown --session ID          delete a session's pod (from its saved state and its name),
                                   then verify
    list                           project pods (named <prefix>-*) that still exist

Needs RUNPOD_API_KEY (shell or .env). Exit codes: 0 ok; 1 setup or a scenario failed (teardown
verified); 2 a pod remains after teardown (check now: it may be billing); 3 not confirmed,
nothing created; 4 preflight failed; 130 interrupted (teardown verified).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from vms.cloud.pod_session import PodPreflight, PodSession, PodSessionState
from vms.cloud.pod_settings import SERVICE, PodSettings, RunpodCredentials
from vms.cloud.remote import SshRemote, generate_key
from vms.cloud.runpod import RunpodApi, RunpodError, urllib_transport
from vms.cloud.session import PreflightError
from vms.common.clock import SystemClock
from vms.common.ids import new_ulid
from vms.common.logs import configure_logging, get_logger
from vms.common.settings import load_settings

log = get_logger(__name__)

STATE_DIR = Path(".gpu-sessions")  # gitignored: per-session key, known_hosts, state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/runpod_session.sh",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="checks that cost nothing")
    run = commands.add_parser("run", help="a whole session")
    run.add_argument("scenarios", nargs="+", help="bench/scenarios/*.yaml (kind model or ladder)")
    run.add_argument("--yes", action="store_true", help="do not ask before creating the pod")
    run.add_argument("--allow-dirty", action="store_true", help="allow uncommitted changes")
    teardown = commands.add_parser("teardown", help="delete a session's pod")
    teardown.add_argument("--session", required=True)
    commands.add_parser("list", help="project pods that still exist")
    return parser


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _describe(pre: PodPreflight, settings: PodSettings) -> str:
    o = pre.offer
    lines = [
        f"  GPU                {settings.gpu_count}x {o.name} ({o.memory_gb} GB), "
        f"{settings.cloud} cloud, stock {o.availability}",
        f"  host CUDA          >= {settings.min_cuda_version}",
        f"  image              {settings.image}",
        f"  expected cost      ${pre.hourly_usd:.3f} per hour (GPU ${o.hourly_usd:.2f} from the "
        f"live catalog + {settings.container_disk_gb} GB container disk), billed per second",
        f"  hard cap           the pod deletes itself after {pre.max_session_minutes} min: "
        f"at most ${pre.max_cost_usd:.2f}",
    ]
    if pre.existing:
        lines.append(f"  WARNING            {len(pre.existing)} project pods still exist:")
        lines += [f"                     {p.id} {p.name} {p.status}" for p in pre.existing]
    return "\n".join(lines)


def _session(settings: PodSettings, api: RunpodApi, session_id: str) -> PodSession:
    t = settings.timeouts
    return PodSession(
        settings,
        api,
        session_id=session_id,
        repo=Path.cwd(),
        state_dir=STATE_DIR,
        clock=SystemClock(),
        sleep=time.sleep,
        remote_factory=lambda host, port, key, known: SshRemote(
            host, settings.ssh_user, key, known, t.ssh_connect_s, t.poll_s, port=port
        ),
        keygen=generate_key,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = load_settings(PodSettings, SERVICE)
    configure_logging(SERVICE, settings.log.level)
    key = RunpodCredentials().api_key
    if key is None:
        _out(
            "RUNPOD_API_KEY is not set: create a key in the Runpod console (Settings > API Keys) "
            "and put it in .env (RUNBOOK, GPU sessions)"
        )
        return 4
    api = RunpodApi(
        urllib_transport(key),
        timeout_s=settings.timeouts.api_s,
        max_attempts=settings.timeouts.api_max_attempts,
    )

    if args.command == "list":
        prefix = f"{settings.name_prefix}-"
        found = [p for p in api.list_pods() if p.name.startswith(prefix)]
        found = [p for p in found if p.status != "TERMINATED"]
        _out("\n".join(f"{p.id} {p.name} {p.status}" for p in found) or "no project pods exist")
        return 0

    if args.command == "teardown":
        session = _session(settings, api, args.session)
        if session.state_path.is_file():
            session.state = PodSessionState.load(session.state_path)
        session.adopt_named()
        report = session.teardown()
        _out("teardown verified" if report.verified else f"LEFTOVERS: {report.leftovers}")
        return 0 if report.verified else 2

    session = _session(settings, api, new_ulid())

    def confirm(pre: PodPreflight) -> bool:
        _out("GPU pod about to be created:\n" + _describe(pre, settings))
        if args.yes:
            return True
        try:
            answer = input("Type 'launch' to start it: ")
        except EOFError:
            return False
        return answer.strip() == "launch"

    try:
        if args.command == "preflight":
            _out("preflight passed:\n" + _describe(session.preflight(), settings))
            return 0
        _out(f"session {session.session_id}")
        code = session.run(
            args.scenarios, confirm, Path("bench/results"), allow_dirty=args.allow_dirty
        )
    except (PreflightError, RunpodError) as exc:  # raised before anything is created
        _out(f"preflight failed: {exc}")
        return 4
    if code == 2:
        _out(
            f"\n!!! TEARDOWN NOT VERIFIED: a pod named {session.name} still exists and may be "
            f"billing.\n!!! Run: scripts/runpod_session.sh teardown --session {session.session_id}"
            "\n"
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
