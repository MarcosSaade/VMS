"""The async run loop: one task per camera, one task per in-flight send.

Paced mode is open loop. Frame k of a camera is due at `phase + k/fps` after the start,
whether or not earlier sends have finished, because a real camera keeps capturing while the
server is slow. Waiting for each response before sending the next (closed loop) would let a slow
server lower the offered load and hide its own latency ("coordinated omission"). A frame due
while the camera already has `max_in_flight_per_camera` sends outstanding is dropped and
counted.

Unpaced mode is closed loop on purpose: every camera sends as fast as its in-flight limit
allows. It measures the maximum rate of the generator plus whatever transport it uses.

Time comes from the event loop (`loop.time()`, monotonic), which is also what `asyncio.sleep`
uses, so tests can run the loop on virtual time. Frame timestamps are wall clock: the run start
read from the injected `Clock`, plus the frame's due offset.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from opentelemetry import propagate, trace

from vms.common.clock import Clock
from vms.common.logs import get_logger
from vms.loadgen.frames import LoadedClip
from vms.loadgen.recorder import Recorder
from vms.loadgen.schedule import CameraPlan, clip_frame, due_time
from vms.loadgen.settings import LoadgenSettings
from vms.loadgen.transport import Outcome, OutgoingFrame, SendResult, Transport

log = get_logger(__name__)


@dataclass(frozen=True)
class RunTiming:
    started_at: str  # wall clock, ISO 8601 UTC
    elapsed_s: float  # schedule start to last camera stopping (excludes the drain)
    drain_s: float  # waiting for in-flight sends after the last camera stopped
    cpu_s: float  # process CPU time over the run (all threads)


class Runner:
    def __init__(
        self,
        settings: LoadgenSettings,
        plans: Sequence[CameraPlan],
        clips: Mapping[str, LoadedClip],
        transport: Transport,
        recorder: Recorder,
        *,
        clock: Clock,
        run_id: str,
        tracer: trace.Tracer | None = None,
        process_time: Callable[[], float] = time.process_time,
    ) -> None:
        self._s = settings
        self._plans = plans
        self._clips = clips
        self._transport = transport
        self._recorder = recorder
        self._clock = clock
        self._run_id = run_id
        self._tracer = tracer or trace.get_tracer(__name__)
        self._process_time = process_time
        self._stop = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()

    def stop(self) -> None:
        """Stop scheduling new frames (graceful shutdown); in-flight sends still complete."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    async def run(self) -> RunTiming:
        loop = asyncio.get_running_loop()
        cpu0 = self._process_time()
        wall0 = self._clock.now()
        t0 = loop.time()
        body = self._paced if self._s.mode == "paced" else self._unpaced
        cameras = [asyncio.create_task(body(plan, t0, wall0)) for plan in self._plans]
        try:
            await asyncio.gather(*cameras)
        finally:
            for task in cameras:
                task.cancel()
        t_end = loop.time()
        await self._drain()
        return RunTiming(
            started_at=wall0.isoformat(),
            elapsed_s=t_end - t0,
            drain_s=loop.time() - t_end,
            cpu_s=self._process_time() - cpu0,
        )

    async def _paced(self, plan: CameraPlan, t0: float, wall0: datetime) -> None:
        loop = asyncio.get_running_loop()
        clip = self._clips[plan.clip_id]
        limit = asyncio.Semaphore(self._s.http.max_in_flight_per_camera)
        k = 0
        while not self._stop.is_set():
            due = due_time(plan, k, self._s.fps)
            if due >= self._s.duration_s:
                return
            delay = t0 + due - loop.time()
            if delay > 0:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                if self._stop.is_set():
                    return
            if limit.locked():
                self._recorder.dropped(due)
            else:
                await limit.acquire()
                lag = loop.time() - (t0 + due)
                self._launch(plan, clip, k, due, lag, wall0, limit)
            k += 1

    async def _unpaced(self, plan: CameraPlan, t0: float, wall0: datetime) -> None:
        loop = asyncio.get_running_loop()
        clip = self._clips[plan.clip_id]
        limit = asyncio.Semaphore(self._s.http.max_in_flight_per_camera)
        k = 0
        while not self._stop.is_set():
            await limit.acquire()
            due = loop.time() - t0
            if due >= self._s.duration_s:
                limit.release()
                return
            self._launch(plan, clip, k, due, 0.0, wall0, limit)
            k += 1

    def _launch(
        self,
        plan: CameraPlan,
        clip: LoadedClip,
        k: int,
        due: float,
        lag: float,
        wall0: datetime,
        limit: asyncio.Semaphore,
    ) -> None:
        index = clip_frame(plan, k, self._s.fps, clip.fps, len(clip.frames))
        ts = wall0 + timedelta(seconds=due)
        frame = OutgoingFrame(
            camera_id=plan.camera_id,
            run_id=self._run_id,
            seq=k,
            ts_unix_ms=int(ts.timestamp() * 1000),
            jpeg=clip.frames[index],
        )
        self._recorder.sent(due, lag)
        task = asyncio.create_task(self._send(frame, due, limit))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _send(self, frame: OutgoingFrame, due: float, limit: asyncio.Semaphore) -> None:
        loop = asyncio.get_running_loop()
        start = loop.time()
        try:
            with self._tracer.start_as_current_span(
                "loadgen.send_frame",
                attributes={"camera_id": frame.camera_id, "seq": frame.seq},
            ) as span:
                headers: dict[str, str] = {}
                propagate.inject(headers)
                if headers:
                    frame = replace(frame, headers=headers)
                result = await self._transport.send(frame)
                span.set_attribute("outcome", result.outcome.value)
        except Exception:
            # A transport must not raise; if one does, count it instead of losing the frame.
            log.exception("send raised", extra={"camera_id": frame.camera_id, "seq": frame.seq})
            result = SendResult(Outcome.FAILED, None, "error")
        finally:
            limit.release()
        self._recorder.completed(due, result, loop.time() - start)

    async def _drain(self) -> None:
        if not self._tasks:
            return
        h = self._s.http
        # Worst case for one send: every attempt times out and every backoff is at its cap.
        timeout = (h.max_retries + 1) * h.request_timeout_s + h.max_retries * h.retry_max_s + 1.0
        _, pending = await asyncio.wait(set(self._tasks), timeout=timeout)
        for task in pending:
            task.cancel()
        if pending:
            log.warning("in-flight sends cancelled at shutdown", extra={"count": len(pending)})
