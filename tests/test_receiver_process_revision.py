from __future__ import annotations

import os

import pytest

from loop_alpha_audit.channel import IDSConfig, ReadObservation, simulate_pool
from loop_alpha_audit.framing import PublicProtocol, encode_file
from loop_alpha_audit.receiver_process import (
    ReceiverActor,
    ReceiverTerminationUnconfirmedError,
)


def test_fresh_process_receives_no_sender_or_channel_truth() -> None:
    public = PublicProtocol(rs_n=7, rs_k=5, mapper="short_core")
    # This variable exists in the parent only. It is not a constructor/request argument.
    private_user = b"parent-only payload"
    frames = encode_file(private_user, public, seed=112)
    output = simulate_pool(
        {frame.strand_index: frame.sequence for frame in frames}, IDSConfig(), coverage=3, seed=224
    )
    with ReceiverActor(public, timeout_seconds=20) as receiver:
        assert receiver.server_metadata["start_method"] == "spawn"
        assert receiver.server_metadata["pid"] != os.getpid()
        result = receiver.receive(output.observations)
        assert receiver.last_request_schema == ("kind", "read_rows", "request_id")
        assert result.status == "OK" and result.user == private_user
        assert receiver.receive(tuple(reversed(output.observations))).user == private_user
        assert receiver.receive(()).status == "RS_FAILED"
        with pytest.raises(TypeError):
            receiver.receive(frames)
        with pytest.raises(TypeError):
            receiver.receive(output.truth)


def test_bad_receiver_parameters_fail_before_start() -> None:
    with pytest.raises(ValueError):
        ReceiverActor(None)
    with pytest.raises(ValueError):
        ReceiverActor(PublicProtocol(rs_n=7, rs_k=5), timeout_seconds=0)


def test_timeout_irreversibly_invalidates_actor_and_no_late_result_leaks() -> None:
    public = PublicProtocol(rs_n=3, rs_k=2, mapper="two_bit")
    frames = encode_file(b"known_file_1", public, seed=2026)
    observations = tuple(ReadObservation(str(i), frame.sequence) for i, frame in enumerate(frames))
    actor = ReceiverActor(public)
    original = actor._connection

    class ForcedPollTimeout:
        def poll(self, timeout):
            return False

        def __getattr__(self, name):
            return getattr(original, name)

    actor._connection = ForcedPollTimeout()
    with pytest.raises(TimeoutError):
        actor.receive(observations)
    assert not actor._process.is_alive()
    assert actor.invalidated_reason is not None
    actor.timeout_seconds = 30
    with pytest.raises(RuntimeError, match="fresh spawn"):
        actor.receive(())
    actor.close()
    with ReceiverActor(public) as fresh:
        assert fresh.receive(()).status == "RS_FAILED"


def test_mismatched_transport_id_invalidates_actor() -> None:
    actor = ReceiverActor(PublicProtocol(rs_n=3, rs_k=2, mapper="two_bit"))
    original = actor._connection

    class WrongResponseID:
        def recv(self):
            response = original.recv()
            return response | {"request_id": response["request_id"] + 1}

        def __getattr__(self, name):
            return getattr(original, name)

    actor._connection = WrongResponseID()
    with pytest.raises(RuntimeError, match="mismatched"):
        actor.receive(())
    assert not actor._process.is_alive()
    with pytest.raises(RuntimeError, match="fresh spawn"):
        actor.receive(())


def test_unconfirmed_termination_has_a_distinct_fatal_type() -> None:
    actor = ReceiverActor.__new__(ReceiverActor)

    class UnconfirmedProcess:
        def is_alive(self):
            return True

        def terminate(self):
            pass

        def join(self, **kwargs):
            pass

        def kill(self):
            pass

    class Connection:
        def close(self):
            pass

    actor._process = UnconfirmedProcess()
    actor._connection = Connection()
    with pytest.raises(ReceiverTerminationUnconfirmedError, match="unconfirmed"):
        actor._invalidate("fault-injected child state")
    assert actor._closed


@pytest.mark.parametrize("failure_at", ["terminate", "join", "kill", "is_alive", "close"])
def test_every_cleanup_api_failure_is_fatal_not_an_ordinary_oserror(failure_at) -> None:
    actor = ReceiverActor.__new__(ReceiverActor)

    class FaultProcess:
        alive = True

        def is_alive(self):
            if failure_at == "is_alive":
                raise OSError("injected is_alive failure")
            return self.alive

        def terminate(self):
            if failure_at == "terminate":
                raise OSError("injected terminate failure")
            if failure_at != "kill":
                self.alive = False

        def join(self, **kwargs):
            if failure_at == "join":
                raise OSError("injected join failure")

        def kill(self):
            if failure_at == "kill":
                raise OSError("injected kill failure")
            self.alive = False

    class FaultConnection:
        def close(self):
            if failure_at == "close":
                raise OSError("injected close failure")

    actor._process = FaultProcess()
    actor._connection = FaultConnection()
    with pytest.raises(ReceiverTerminationUnconfirmedError):
        actor._invalidate("unit cleanup fault")
    assert actor._closed


def test_remote_pipe_cleanup_after_start_cannot_escape_as_recoverable(monkeypatch) -> None:
    import loop_alpha_audit.receiver_process as boundary

    class LocalConnection:
        def close(self):
            pass

    class RemoteConnection:
        def close(self):
            raise OSError("injected remote endpoint cleanup failure")

    class FakeProcess:
        alive = True

        def start(self):
            pass

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive = False

        def join(self, **kwargs):
            pass

    class Context:
        def Pipe(self):
            return LocalConnection(), RemoteConnection()

        def Process(self, **kwargs):
            return FakeProcess()

    monkeypatch.setattr(boundary.mp, "get_context", lambda method: Context())
    with pytest.raises(ReceiverTerminationUnconfirmedError, match="remote endpoint cleanup"):
        ReceiverActor(PublicProtocol(rs_n=3, rs_k=2))
