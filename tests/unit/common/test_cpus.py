from pathlib import Path

from vms.common.cpus import cpu_quota


def test_cgroup_v2_quota_in_whole_cpus(tmp_path: Path) -> None:
    v2 = tmp_path / "cpu.max"
    v2.write_text("900000 100000\n")
    assert cpu_quota(v2) == 9
    v2.write_text("150000 100000\n")
    assert cpu_quota(v2) == 1
    v2.write_text("50000 100000\n")
    assert cpu_quota(v2) == 1  # at least one
    v2.write_text("max 100000\n")
    assert cpu_quota(v2) is None


def test_cgroup_v1_and_no_cgroup(tmp_path: Path) -> None:
    missing = tmp_path / "none"
    q, p = tmp_path / "quota", tmp_path / "period"
    q.write_text("400000")
    p.write_text("100000")
    assert cpu_quota(missing, q, p) == 4
    q.write_text("-1")
    assert cpu_quota(missing, q, p) is None
    assert cpu_quota(missing, missing, missing) is None
