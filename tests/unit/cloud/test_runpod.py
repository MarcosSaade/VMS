import json
import random
import urllib.error
from typing import Any

import pytest

from vms.cloud.runpod import PodSpec, RunpodApi, RunpodError, parse_pod

POD = {
    "id": "pod-1",
    "name": "vms-gpu-01s",
    "status": "RUNNING",
    "cost": 0.59,
    "dataCenterId": "EU-RO-1",
    "cudaVersion": "13.2",
    "createdAt": "2026-10-19T12:00:00Z",
    "ssh": {
        "proxy": {"host": "ssh.runpod.io", "port": 22, "username": "x", "command": "ssh x"},
        "direct": {"host": "203.0.113.9", "port": 40122, "username": "root", "command": "ssh"},
    },
}
L4 = {
    "id": "NVIDIA L4",
    "name": "L4",
    "memory": 24,
    "availability": "LOW",
    "price": {"community": 0.44, "secure": 0.59},
    "dataCenters": [{"id": "EU-RO-1", "availability": "LOW"}],
}


class FakeTransport:
    """Answers queued (status, payload) pairs, or raises queued exceptions, in order."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, str, Any]] = []

    def __call__(self, method: str, url: str, body: Any, timeout_s: float) -> tuple[int, Any]:
        self.calls.append((method, url, body))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        assert isinstance(answer, tuple)
        return answer


def api(transport: FakeTransport, attempts: int = 3) -> tuple[RunpodApi, list[float]]:
    sleeps: list[float] = []
    client = RunpodApi(
        transport,
        timeout_s=5,
        max_attempts=attempts,
        base_url="https://api.test/v2",
        sleep=sleeps.append,
        rng=random.Random(1),
    )
    return client, sleeps


def test_pod_parsing_takes_direct_ssh_not_the_proxy() -> None:
    pod = parse_pod(POD)
    assert pod.ssh is not None
    assert (pod.ssh.host, pod.ssh.port, pod.ssh.user) == ("203.0.113.9", 40122, "root")
    assert pod.created_at is not None
    assert pod.created_at.isoformat() == "2026-10-19T12:00:00+00:00"
    provisioning = parse_pod({"id": "p", "status": "PROVISIONING", "ssh": {"direct": None}})
    assert provisioning.ssh is None
    assert provisioning.name == ""


def test_gpu_offer_reads_the_price_for_the_requested_cloud() -> None:
    t = FakeTransport((200, L4))
    offer = api(t)[0].gpu_offer("NVIDIA L4", "SECURE", "12.8")
    assert (offer.hourly_usd, offer.availability, offer.data_centers) == (0.59, "LOW", ["EU-RO-1"])
    method, url, _ = t.calls[0]
    assert method == "GET"
    assert url.startswith("https://api.test/v2/catalog/gpus/NVIDIA%20L4?")
    for part in ("include=AVAILABILITY", "product=POD", "cloud=SECURE", "minCudaVersion=12.8"):
        assert part in url


def test_gpu_offer_errors_on_an_unknown_type_or_a_missing_price() -> None:
    with pytest.raises(RunpodError, match="unknown GPU type"):
        api(FakeTransport((404, {"error": "not found"})))[0].gpu_offer("H900", "SECURE", "12.8")
    community_only = {**L4, "price": {"community": 0.44, "secure": 0}}
    with pytest.raises(RunpodError, match="no SECURE price"):
        api(FakeTransport((200, community_only)))[0].gpu_offer("NVIDIA L4", "SECURE", "12.8")


def test_auth_failures_name_the_key() -> None:
    with pytest.raises(RunpodError, match="RUNPOD_API_KEY"):
        api(FakeTransport((401, {"error": "unauthorized"})))[0].list_pods()


def test_reads_retry_429_5xx_and_connection_errors_with_bounded_jittered_backoff() -> None:
    t = FakeTransport((429, None), urllib.error.URLError("reset"), (200, {"pods": []}))
    client, sleeps = api(t, attempts=3)
    assert client.list_pods() == []
    assert len(t.calls) == 3
    assert len(sleeps) == 2
    assert all(0 <= s <= 30 for s in sleeps)
    exhausted = FakeTransport((503, "busy"), (503, "busy"), (503, "busy"))
    with pytest.raises(RunpodError, match="503"):
        api(exhausted, attempts=3)[0].list_pods()
    assert len(exhausted.calls) == 3


def test_create_is_never_retried() -> None:
    spec = PodSpec(
        name="vms-gpu-01s",
        image="img@sha256:1",
        gpu_type="NVIDIA L4",
        gpu_count=1,
        cloud="SECURE",
        min_cuda_version="12.8",
        container_disk_gb=40,
        ports=["22/tcp"],
        env={"PUBLIC_KEY": "ssh-ed25519 AAAA"},
        cmd=["bash", "-c", "exec /start.sh"],
    )
    t = FakeTransport((503, "busy"))
    with pytest.raises(RunpodError, match="503"):
        api(t)[0].create_pod(spec)
    assert len(t.calls) == 1
    body = t.calls[0][2]
    assert body["gpu"] == {"id": "NVIDIA L4", "count": 1, "minCudaVersion": "12.8"}
    assert "mounts" not in body  # no volume: nothing outlives the pod
    assert "dataCenterIds" not in body
    assert json.loads(json.dumps(body)) == body


def test_list_follows_the_cursor() -> None:
    t = FakeTransport(
        (200, {"pods": [POD], "pagination": {"hasNextPage": True, "nextCursor": "c/2"}}),
        (200, {"pods": [{**POD, "id": "pod-2"}], "pagination": {"hasNextPage": False}}),
    )
    assert [p.id for p in api(t)[0].list_pods()] == ["pod-1", "pod-2"]
    assert t.calls[1][1] == "https://api.test/v2/pods?cursor=c/2"


def test_get_and_delete_treat_404_as_gone() -> None:
    client = api(FakeTransport((404, None), (404, None), (204, None)))[0]
    assert client.get_pod("pod-1") is None
    client.delete_pod("pod-1")
    client.delete_pod("pod-1")
    with pytest.raises(RunpodError, match="409"):
        api(FakeTransport((409, {"error": "locked"})))[0].delete_pod("pod-1")
