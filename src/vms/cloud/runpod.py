"""The Runpod calls a pod session makes, behind a small interface (`Pods`) so tests use a fake.

`RunpodApi` is the real implementation on Runpod's REST v2 API (`https://api.runpod.io/v2`,
OpenAPI at `/v2/openapi.json`): the GPU catalog (price and stock), pods (create, read, list,
delete). Every request has a timeout; reads and deletes are retried on 429, 5xx and connection
errors with exponential backoff and full jitter, a bounded number of times. A create is never
retried: a create that timed out may still have made a pod, which the session finds by its name.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from pydantic import SecretStr

API_BASE = "https://api.runpod.io/v2"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class RunpodError(Exception):
    """A Runpod API call failed in a way the session cannot recover from."""


@dataclass(frozen=True)
class GpuOffer:
    """One GPU type in the catalog, priced for the requested cloud tier."""

    id: str
    name: str
    memory_gb: int
    hourly_usd: float
    availability: str  # NONE, LOW, MEDIUM, HIGH (stock for pods, this cloud and CUDA floor)
    data_centers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PodSpec:
    name: str
    image: str
    gpu_type: str
    gpu_count: int
    cloud: str
    min_cuda_version: str
    container_disk_gb: int
    ports: list[str]
    env: dict[str, str]
    cmd: list[str]
    data_center_ids: list[str] = field(default_factory=list)

    def body(self) -> dict[str, Any]:
        """The `POST /v2/pods` body. No `mounts`: no volume, nothing that outlives the pod."""
        body: dict[str, Any] = {
            "name": self.name,
            "image": self.image,
            "gpu": {
                "id": self.gpu_type,
                "count": self.gpu_count,
                "minCudaVersion": self.min_cuda_version,
            },
            "cloud": self.cloud,
            "disk": self.container_disk_gb,
            "ports": self.ports,
            "env": self.env,
            "cmd": self.cmd,
        }
        if self.data_center_ids:
            body["dataCenterIds"] = self.data_center_ids
        return body


@dataclass(frozen=True)
class SshEndpoint:
    host: str
    port: int
    user: str


@dataclass(frozen=True)
class Pod:
    id: str
    name: str
    status: str  # PROVISIONING, STARTING, RUNNING, EXITED, ERROR, TERMINATED
    hourly_usd: float = 0.0
    data_center: str | None = None
    cuda_version: str | None = None
    ssh: SshEndpoint | None = None  # direct SSH over the published 22/tcp, once running
    created_at: datetime | None = None
    ports: tuple[str, ...] = ()  # runtime port mappings, "private->ip:public/type"
    vcpus: float | None = None  # allocated to the pod (gpu.vcpuCount)
    memory_gb: float | None = None


class Pods(Protocol):
    def gpu_offer(self, gpu_type: str, cloud: str, min_cuda_version: str) -> GpuOffer: ...
    def create_pod(self, spec: PodSpec) -> Pod: ...
    def get_pod(self, pod_id: str) -> Pod | None:
        """None when the pod does not exist (any more)."""
        ...

    def list_pods(self) -> list[Pod]: ...
    def delete_pod(self, pod_id: str) -> None:
        """Terminate and delete; a pod already gone is not an error."""
        ...


Transport = Callable[[str, str, dict[str, Any] | None, float], tuple[int, Any]]


def urllib_transport(api_key: SecretStr) -> Transport:
    """(method, url, json body, timeout) -> (status, parsed JSON or None)."""

    def call(method: str, url: str, body: dict[str, Any] | None, timeout_s: float) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {api_key.get_secret_value()}")
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read()
                return resp.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw) if raw else None
            except ValueError:
                payload = raw.decode(errors="replace")[:2000]
            return exc.code, payload

    return call


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_pod(item: dict[str, Any]) -> Pod:
    direct = (item.get("ssh") or {}).get("direct")
    ssh = (
        SshEndpoint(host=direct["host"], port=int(direct["port"]), user=direct["username"])
        if direct
        else None
    )
    return Pod(
        id=item["id"],
        name=item.get("name") or "",
        status=item.get("status") or "",
        hourly_usd=float(item.get("cost") or 0.0),
        data_center=item.get("dataCenterId"),
        cuda_version=item.get("cudaVersion"),
        ssh=ssh,
        created_at=_parse_time(item.get("createdAt")),
        vcpus=(item.get("gpu") or item.get("cpu") or {}).get("vcpuCount"),
        memory_gb=(item.get("gpu") or item.get("cpu") or {}).get("memory"),
        ports=tuple(
            f"{p.get('private')}->{p.get('ip')}:{p.get('public')}/{p.get('type')}"
            for p in (item.get("runtime") or {}).get("ports") or []
        ),
    )


class RunpodApi:
    """`Pods` on Runpod's REST v2 API."""

    def __init__(
        self,
        transport: Transport,
        *,
        timeout_s: float,
        max_attempts: int,
        base_url: str = API_BASE,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._transport = transport
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._base = base_url.rstrip("/")
        self._sleep = sleep
        self._rng = rng or random.Random()

    def _call(
        self, method: str, path: str, body: dict[str, Any] | None = None, *, retry: bool = True
    ) -> tuple[int, Any]:
        url = f"{self._base}{path}"
        attempts = self._max_attempts if retry else 1
        for attempt in range(1, attempts + 1):
            try:
                status, payload = self._transport(method, url, body, self._timeout_s)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == attempts:
                    raise RunpodError(f"{method} {path}: {exc}") from exc
            else:
                if status not in RETRY_STATUSES or attempt == attempts:
                    return status, payload
            # Exponential backoff with full jitter, capped at 30 s.
            self._sleep(self._rng.uniform(0, min(30.0, 2.0**attempt)))
        raise AssertionError("unreachable")

    @staticmethod
    def _check(method: str, path: str, status: int, payload: Any) -> Any:
        if 200 <= status < 300:
            return payload
        if status in (401, 403):
            raise RunpodError(
                f"{method} {path}: {status} (check RUNPOD_API_KEY and its permissions): {payload}"
            )
        raise RunpodError(f"{method} {path}: {status}: {payload}")

    def gpu_offer(self, gpu_type: str, cloud: str, min_cuda_version: str) -> GpuOffer:
        query = urllib.parse.urlencode(
            {
                "include": "AVAILABILITY",
                "product": "POD",
                "cloud": cloud,
                "minCudaVersion": min_cuda_version,
            }
        )
        path = f"/catalog/gpus/{urllib.parse.quote(gpu_type, safe='')}?{query}"
        status, payload = self._call("GET", path)
        if status == 404:
            raise RunpodError(f"unknown GPU type {gpu_type!r} (GET /v2/catalog/gpus lists them)")
        item = self._check("GET", path, status, payload)
        price = float((item.get("price") or {}).get(cloud.lower()) or 0.0)
        if price <= 0:
            raise RunpodError(f"{gpu_type} has no {cloud} price in the catalog")
        return GpuOffer(
            id=item["id"],
            name=item.get("name") or item["id"],
            memory_gb=int(item.get("memory") or 0),
            hourly_usd=price,
            availability=str(item.get("availability") or "NONE"),
            data_centers=[dc["id"] for dc in item.get("dataCenters") or []],
        )

    def create_pod(self, spec: PodSpec) -> Pod:
        status, payload = self._call("POST", "/pods", spec.body(), retry=False)
        return parse_pod(self._check("POST", "/pods", status, payload))

    def get_pod(self, pod_id: str) -> Pod | None:
        path = f"/pods/{urllib.parse.quote(pod_id, safe='')}"
        status, payload = self._call("GET", path)
        if status == 404:
            return None
        return parse_pod(self._check("GET", path, status, payload))

    def list_pods(self) -> list[Pod]:
        pods: list[Pod] = []
        cursor: str | None = None
        while True:
            path = "/pods" + (f"?cursor={urllib.parse.quote(cursor)}" if cursor else "")
            status, payload = self._call("GET", path)
            page = self._check("GET", path, status, payload)
            pods += [parse_pod(item) for item in page.get("pods") or []]
            pagination = page.get("pagination") or {}
            cursor = pagination.get("nextCursor")
            if not pagination.get("hasNextPage") or not cursor:
                return pods

    def delete_pod(self, pod_id: str) -> None:
        path = f"/pods/{urllib.parse.quote(pod_id, safe='')}"
        status, payload = self._call("DELETE", path)
        if status == 404:
            return
        self._check("DELETE", path, status, payload)
