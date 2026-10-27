"""Fresh-process receiver boundary: only public parameters and observed reads.

`spawn`, not `fork`, prevents a child from inheriting the sender/channel memory.
This is process-memory isolation, not an OS filesystem sandbox. The reviewed
receiver does not read files, channel truth, expected payloads, or source labels.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from multiprocessing.connection import Connection
from typing import Sequence

from .channel import ReadObservation
from .framing import PublicProtocol
from .receiver import PoolReconstruction, reconstruct_pool


class ReceiverTerminationUnconfirmedError(RuntimeError):
    """Fatal: the experiment must not create another receiver after this error."""


def _serve_receiver(protocol: PublicProtocol, connection: Connection) -> None:
    connection.send({"kind": "ready", "pid": os.getpid(), "start_method": mp.get_start_method()})
    last_request_id = 0
    try:
        while True:
            request = connection.recv()
            if request == {"kind": "stop"}:
                break
            if (
                not isinstance(request, dict)
                or set(request) != {"kind", "request_id", "read_rows"}
                or request["kind"] != "reads"
                or type(request["request_id"]) is not int
                or request["request_id"] != last_request_id + 1
            ):
                connection.send(
                    {
                        "kind": "error",
                        "request_id": None,
                        "reason": "invalid receiver request schema or transport ID",
                    }
                )
                continue
            last_request_id = request["request_id"]
            try:
                rows = request["read_rows"]
                if not isinstance(rows, (tuple, list)) or any(
                    not isinstance(row, (tuple, list))
                    or len(row) != 2
                    or not all(isinstance(value, str) for value in row)
                    for row in rows
                ):
                    raise ValueError("read rows must contain only opaque IDs and observed bases")
                reads = tuple(
                    ReadObservation(identifier, sequence) for identifier, sequence in rows
                )
                result = reconstruct_pool(reads, protocol)
                connection.send({"kind": "result", "request_id": last_request_id, "result": result})
            except Exception as exc:
                connection.send(
                    {
                        "kind": "error",
                        "request_id": last_request_id,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    except EOFError:
        pass
    finally:
        connection.close()


class ReceiverActor:
    """Persistent spawned receiver, amortizing process startup across files."""

    def __init__(self, protocol: PublicProtocol, *, timeout_seconds: float = 120):
        if not isinstance(protocol, PublicProtocol) or not 0 < timeout_seconds <= 600:
            raise ValueError("public protocol and bounded positive timeout required")
        self.timeout_seconds = timeout_seconds
        self._closed = False
        self._request_counter = 0
        self.invalidated_reason: str | None = None
        context = mp.get_context("spawn")
        self._connection, remote = context.Pipe()
        self._process = context.Process(target=_serve_receiver, args=(protocol, remote))
        try:
            self._process.start()
        except BaseException as exc:
            # Process.start may fail before a trustworthy child handle exists.
            # Conservatively halt, rather than assert that no child was created.
            self._closed = True
            for connection in (remote, self._connection):
                try:
                    connection.close()
                except BaseException:
                    pass  # A dedicated fatal error is raised regardless.
            raise ReceiverTerminationUnconfirmedError(
                "receiver process start failed; child termination cannot be certified"
            ) from exc
        try:
            remote.close()
        except BaseException as exc:
            self._invalidate("remote pipe endpoint failed to close after child start")
            raise ReceiverTerminationUnconfirmedError(
                "remote endpoint cleanup failed after child start; experiment must halt"
            ) from exc
        try:
            if not self._connection.poll(timeout_seconds):
                raise TimeoutError("spawned receiver did not initialize within the declared limit")
            self.server_metadata = self._connection.recv()
            if (
                not isinstance(self.server_metadata, dict)
                or self.server_metadata.get("kind") != "ready"
                or self.server_metadata.get("start_method") != "spawn"
            ):
                raise RuntimeError("receiver did not confirm a fresh spawned process")
        except BaseException as exc:
            self._invalidate(str(exc))
            raise
        self.last_request_schema: tuple[str, ...] | None = None

    def receive(self, reads: Sequence[ReadObservation]) -> PoolReconstruction:
        if self._closed:
            raise RuntimeError("receiver actor is closed/invalidated; a fresh spawn is required")
        if any(type(read) is not ReadObservation for read in reads):
            raise TypeError("sender records or truth objects cannot cross the receiver boundary")
        self._request_counter += 1
        request_id = self._request_counter  # Transport sequence only, never a source/file label.
        request = {
            "kind": "reads",
            "request_id": request_id,
            "read_rows": tuple((read.read_id, read.sequence) for read in reads),
        }
        self.last_request_schema = tuple(sorted(request))
        try:
            self._connection.send(request)
            if not self._connection.poll(self.timeout_seconds):
                raise TimeoutError("receiver exceeded the declared per-file timeout")
            response = self._connection.recv()
            if (
                not isinstance(response, dict)
                or response.get("request_id") != request_id
                or response.get("kind") != "result"
                or type(response.get("result")) is not PoolReconstruction
            ):
                reason = (
                    response.get("reason", "mismatched/invalid receiver response")
                    if isinstance(response, dict)
                    else "invalid receiver response type"
                )
                raise RuntimeError(reason)
        except (TimeoutError, EOFError, OSError, RuntimeError) as exc:
            self._invalidate(str(exc))
            raise
        return response["result"]

    def _invalidate(self, reason: str) -> None:
        """Irreversible failure. A delayed result cannot be consumed by a later file."""
        self.invalidated_reason = reason
        self._closed = True
        try:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
            if self._process.is_alive():
                raise ReceiverTerminationUnconfirmedError("child remains live after forced cleanup")
            self._connection.close()
        except BaseException as exc:
            # A failed terminate/join/kill/is_alive/close operation cannot
            # escape as a routine recoverable OSError, especially during
            # construction before the caller has received the actor handle.
            try:
                self._connection.close()
            except BaseException:
                pass
            raise ReceiverTerminationUnconfirmedError(
                "receiver termination/cleanup is unconfirmed; do not restart the experiment"
            ) from exc

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            if hasattr(self, "_process") and self._process.is_alive():
                try:
                    self._connection.send({"kind": "stop"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self._process.join(timeout=2)
                if self._process.is_alive():
                    self._invalidate("receiver failed to exit after stop; forced cleanup")
            if hasattr(self, "_connection"):
                self._connection.close()
        except ReceiverTerminationUnconfirmedError:
            raise
        except BaseException as exc:
            self._invalidate("graceful receiver cleanup raised an exception")
            raise ReceiverTerminationUnconfirmedError(
                "receiver cleanup raised an exception; experiment must halt"
            ) from exc

    def __enter__(self) -> ReceiverActor:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
