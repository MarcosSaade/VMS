"""How many CPUs this process may actually use.

`os.cpu_count()` reports the host's CPUs. Inside a container with a CPU quota (a Runpod pod
gets a share of a 96-thread host) thread pools sized from it (ONNX Runtime, OpenCV, PyTorch)
start far more threads than the quota allows, and the kernel throttles them in bursts, which
shows up as erratic latency (E-064). `cpu_quota` reads the cgroup limit; `None` means no
quota (a laptop, an EC2 instance, a container without a CPU limit).
"""

from __future__ import annotations

import math
from pathlib import Path

CGROUP_V2 = Path("/sys/fs/cgroup/cpu.max")
CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def cpu_quota(
    v2: Path = CGROUP_V2, v1_quota: Path = CGROUP_V1_QUOTA, v1_period: Path = CGROUP_V1_PERIOD
) -> int | None:
    """Whole CPUs allowed by the cgroup quota (at least 1), or None without a quota."""
    raw = _read(v2)
    if raw is not None:
        quota, _, period = raw.partition(" ")
        if quota == "max" or not period:
            return None
        return max(1, math.floor(int(quota) / int(period)))
    quota_s, period_s = _read(v1_quota), _read(v1_period)
    if quota_s is None or period_s is None or int(quota_s) <= 0:
        return None
    return max(1, math.floor(int(quota_s) / int(period_s)))
