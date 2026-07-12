import asyncio
import os

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.session_context import (
    async_delivery_supported,
    clear_session_vars,
    get_session_env,
    reset_session_vars,
    scoped_session_vars,
    set_session_vars,
)


@pytest.fixture(autouse=True)
def _reset_contextvars():
    """Reset all session contextvars to _UNSET between tests.

    In production each asyncio.Task gets a fresh context copy where the
    defaults are _UNSET.  In tests all functions share the same thread
    context, so a clear_session_vars() from test A (which sets vars to "")
    would leak into test B.  This fixture ensures each test starts clean.
    """
    reset_session_vars()
    yield
    reset_session_vars()


def _session_snapshot():
    names = (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_SOURCE",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_CHAT_NAME",
        "HERMES_SESSION_THREAD_ID",
        "HERMES_SESSION_USER_ID",
        "HERMES_SESSION_USER_NAME",
        "HERMES_SESSION_KEY",
        "HERMES_SESSION_ID",
        "HERMES_UI_SESSION_ID",
        "HERMES_SESSION_MESSAGE_ID",
        "HERMES_SESSION_PROFILE",
    )
    return {name: get_session_env(name) for name in names}


def test_scoped_session_vars_restores_nested_context_and_cwd(tmp_path):
    """A nested scope restores every field, async capability, and runtime CWD."""
    from agent.runtime_cwd import resolve_agent_cwd

    outer_cwd = tmp_path / "outer"
    inner_cwd = tmp_path / "inner"
    outer_cwd.mkdir()
    inner_cwd.mkdir()
    outer = {
        "platform": "outer-platform",
        "source": "outer-source",
        "chat_id": "outer-chat",
        "chat_name": "outer-name",
        "thread_id": "outer-thread",
        "user_id": "outer-user",
        "user_name": "outer-user-name",
        "session_key": "outer-key",
        "session_id": "outer-session",
        "ui_session_id": "outer-ui",
        "message_id": "outer-message",
        "profile": "outer-profile",
        "cwd": str(outer_cwd),
        "async_delivery": False,
    }
    inner = {
        key: f"inner-{key}" for key in outer if key not in {"cwd", "async_delivery"}
    }
    inner.update(cwd=str(inner_cwd), async_delivery=True)

    with scoped_session_vars(**outer):
        outer_snapshot = _session_snapshot()
        assert async_delivery_supported() is False
        assert resolve_agent_cwd() == outer_cwd

        with scoped_session_vars(**inner):
            assert get_session_env("HERMES_SESSION_CHAT_ID") == "inner-chat_id"
            assert get_session_env("HERMES_UI_SESSION_ID") == "inner-ui_session_id"
            assert async_delivery_supported() is True
            assert resolve_agent_cwd() == inner_cwd

        assert _session_snapshot() == outer_snapshot
        assert async_delivery_supported() is False
        assert resolve_agent_cwd() == outer_cwd


def test_scoped_session_vars_restores_after_exception(tmp_path):
    """Handler failures cannot destroy the enclosing session or runtime CWD."""
    from agent.runtime_cwd import resolve_agent_cwd

    outer_cwd = tmp_path / "outer"
    inner_cwd = tmp_path / "inner"
    outer_cwd.mkdir()
    inner_cwd.mkdir()

    with scoped_session_vars(
        platform="outer",
        chat_id="outer-chat",
        cwd=str(outer_cwd),
        async_delivery=False,
    ):
        before = _session_snapshot()
        with pytest.raises(RuntimeError, match="plugin failed"):
            with scoped_session_vars(
                platform="inner",
                chat_id="inner-chat",
                cwd=str(inner_cwd),
                async_delivery=True,
            ):
                raise RuntimeError("plugin failed")

        assert _session_snapshot() == before
        assert async_delivery_supported() is False
        assert resolve_agent_cwd() == outer_cwd


@pytest.mark.asyncio
async def test_scoped_session_vars_isolates_concurrent_async_handlers():
    """Overlapping async plugin scopes retain their own complete session state."""
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = 0
    results = {}

    async def handler(label):
        nonlocal started
        with scoped_session_vars(
            platform="telegram",
            chat_id=f"chat-{label}",
            thread_id=f"thread-{label}",
            session_key=f"session-{label}",
            async_delivery=(label == "a"),
        ):
            started += 1
            if started == 2:
                both_started.set()
            await both_started.wait()
            await release.wait()
            results[label] = (
                get_session_env("HERMES_SESSION_CHAT_ID"),
                get_session_env("HERMES_SESSION_THREAD_ID"),
                get_session_env("HERMES_SESSION_KEY"),
                async_delivery_supported(),
            )

    task_a = asyncio.create_task(handler("a"))
    task_b = asyncio.create_task(handler("b"))
    await both_started.wait()
    release.set()
    await asyncio.gather(task_a, task_b)

    assert results == {
        "a": ("chat-a", "thread-a", "session-a", True),
        "b": ("chat-b", "thread-b", "session-b", False),
    }


def test_set_session_env_sets_contextvars(monkeypatch):
    """_set_session_env should populate contextvars, not os.environ."""
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name="Group",
        chat_type="group",
        user_id="123456",
        user_name="alice",
        thread_id="17585",
    )
    context = SessionContext(source=source, connected_platforms=[], home_channels={})

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)

    tokens = runner._set_session_env(context)

    # Values should be readable via get_session_env (contextvar path)
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"
    assert get_session_env("HERMES_SESSION_SOURCE") == ""
    assert get_session_env("HERMES_SESSION_CHAT_ID") == "-1001"
    assert get_session_env("HERMES_SESSION_CHAT_NAME") == "Group"
    assert get_session_env("HERMES_SESSION_USER_ID") == "123456"
    assert get_session_env("HERMES_SESSION_USER_NAME") == "alice"
    assert get_session_env("HERMES_SESSION_THREAD_ID") == "17585"

    # os.environ should NOT be touched
    assert os.getenv("HERMES_SESSION_PLATFORM") is None
    assert os.getenv("HERMES_SESSION_SOURCE") is None
    assert os.getenv("HERMES_SESSION_THREAD_ID") is None

    # Clean up
    runner._clear_session_env(tokens)


def test_session_source_uses_contextvars(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    tokens = set_session_vars(source="tool")

    assert get_session_env("HERMES_SESSION_SOURCE") == "tool"

    clear_session_vars(tokens)

    assert get_session_env("HERMES_SESSION_SOURCE") == ""


def test_clear_session_env_restores_previous_state(monkeypatch):
    """_clear_session_env should restore contextvars to their pre-handler values."""
    runner = object.__new__(GatewayRunner)

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name="Group",
        chat_type="group",
        user_id="123456",
        user_name="alice",
        thread_id="17585",
    )
    context = SessionContext(source=source, connected_platforms=[], home_channels={})

    tokens = runner._set_session_env(context)
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"
    assert get_session_env("HERMES_SESSION_USER_ID") == "123456"

    runner._clear_session_env(tokens)

    # After clear, contextvars should return to defaults (empty)
    assert get_session_env("HERMES_SESSION_PLATFORM") == ""
    assert get_session_env("HERMES_SESSION_CHAT_ID") == ""
    assert get_session_env("HERMES_SESSION_CHAT_NAME") == ""
    assert get_session_env("HERMES_SESSION_USER_ID") == ""
    assert get_session_env("HERMES_SESSION_USER_NAME") == ""
    assert get_session_env("HERMES_SESSION_THREAD_ID") == ""


def test_get_session_env_falls_back_to_os_environ(monkeypatch):
    """get_session_env should fall back to os.environ when contextvar is unset."""
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")

    # No contextvar set — should read from os.environ
    assert get_session_env("HERMES_SESSION_PLATFORM") == "discord"

    # Now set a contextvar — should prefer it
    tokens = set_session_vars(platform="telegram")
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"

    # After clear — should return "" (explicitly cleared), NOT fall back
    # to os.environ.  This is the fix for #10304: stale os.environ values
    # must not leak through after a gateway session is cleaned up.
    clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_PLATFORM") == ""


def test_get_session_env_default_when_nothing_set(monkeypatch):
    """get_session_env returns default when neither contextvar nor env is set."""
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)

    assert get_session_env("HERMES_SESSION_PLATFORM") == ""
    assert get_session_env("HERMES_SESSION_PLATFORM", "fallback") == "fallback"


def test_set_session_env_handles_missing_optional_fields():
    """_set_session_env should handle None chat_name and thread_id gracefully."""
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name=None,
        chat_type="private",
        thread_id=None,
    )
    context = SessionContext(source=source, connected_platforms=[], home_channels={})

    tokens = runner._set_session_env(context)

    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"
    assert get_session_env("HERMES_SESSION_CHAT_ID") == "-1001"
    assert get_session_env("HERMES_SESSION_CHAT_NAME") == ""
    assert get_session_env("HERMES_SESSION_THREAD_ID") == ""

    runner._clear_session_env(tokens)


# ---------------------------------------------------------------------------
# SESSION_KEY contextvars tests
# ---------------------------------------------------------------------------


def test_session_key_set_via_contextvars(monkeypatch):
    """set_session_vars should set HERMES_SESSION_KEY via contextvars."""
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

    tokens = set_session_vars(
        platform="telegram",
        chat_id="-1001",
        session_key="tg:-1001:17585",
    )
    assert get_session_env("HERMES_SESSION_KEY") == "tg:-1001:17585"

    clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_KEY") == ""


def test_session_key_falls_back_to_os_environ(monkeypatch):
    """get_session_env for SESSION_KEY should fall back to os.environ."""
    monkeypatch.setenv("HERMES_SESSION_KEY", "env-session-123")

    # No contextvar set — should read from os.environ
    assert get_session_env("HERMES_SESSION_KEY") == "env-session-123"

    # Set contextvar — should prefer it
    tokens = set_session_vars(session_key="ctx-session-456")
    assert get_session_env("HERMES_SESSION_KEY") == "ctx-session-456"

    # After clear — should return "" (explicitly cleared), not os.environ (#10304)
    clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_KEY") == ""


def test_session_id_set_via_contextvars(monkeypatch):
    """set_session_vars should set HERMES_SESSION_ID via contextvars."""
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-env-session")

    tokens = set_session_vars(session_id="ctx-session-456")
    assert get_session_env("HERMES_SESSION_ID") == "ctx-session-456"

    clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_ID") == ""


def test_set_session_env_includes_session_key():
    """_set_session_env should propagate session_key from SessionContext."""
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name="Group",
        chat_type="group",
        thread_id="17585",
    )
    context = SessionContext(
        source=source,
        connected_platforms=[],
        home_channels={},
        session_key="tg:-1001:17585",
    )

    # Capture baseline value before setting (may be non-empty from another
    # test in the same pytest-xdist worker sharing the context).
    tokens = runner._set_session_env(context)
    assert get_session_env("HERMES_SESSION_KEY") == "tg:-1001:17585"
    runner._clear_session_env(tokens)
    # After clearing, the session key must not retain the value we just set.
    # The exact post-clear value depends on context propagation from other
    # tests, so only check that our value was removed, not what replaced it.
    assert get_session_env("HERMES_SESSION_KEY") != "tg:-1001:17585"


def test_session_key_no_race_condition_with_contextvars(monkeypatch):
    """Prove contextvars isolates SESSION_KEY across concurrent async tasks.

    Two tasks set different session keys. With contextvars each task
    reads back its own value. With os.environ the second task would
    overwrite the first (the old bug).
    """
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

    results = {}

    async def handler(key: str, delay: float):
        tokens = set_session_vars(session_key=key)
        try:
            await asyncio.sleep(delay)
            read_back = get_session_env("HERMES_SESSION_KEY")
            results[key] = read_back
        finally:
            clear_session_vars(tokens)

    async def run():
        task_a = asyncio.create_task(handler("session-A", 0.15))
        await asyncio.sleep(0.05)
        task_b = asyncio.create_task(handler("session-B", 0.05))
        await asyncio.gather(task_a, task_b)

    asyncio.run(run())

    # Both tasks must read back their own session key
    assert results["session-A"] == "session-A", (
        f"Session A got '{results['session-A']}' instead of 'session-A' — race condition!"
    )
    assert results["session-B"] == "session-B", (
        f"Session B got '{results['session-B']}' instead of 'session-B' — race condition!"
    )


@pytest.mark.asyncio
async def test_run_in_executor_with_context_preserves_session_env(monkeypatch):
    """Gateway executor work should inherit session contextvars for tool routing."""
    runner = object.__new__(GatewayRunner)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="2144471399",
        chat_type="dm",
        user_id="123456",
        user_name="alice",
        thread_id=None,
    )
    context = SessionContext(
        source=source,
        connected_platforms=[],
        home_channels={},
        session_key="agent:main:telegram:dm:2144471399",
    )

    tokens = runner._set_session_env(context)
    try:
        result = await runner._run_in_executor_with_context(
            lambda: {
                "platform": get_session_env("HERMES_SESSION_PLATFORM"),
                "chat_id": get_session_env("HERMES_SESSION_CHAT_ID"),
                "user_id": get_session_env("HERMES_SESSION_USER_ID"),
                "session_key": get_session_env("HERMES_SESSION_KEY"),
            }
        )
    finally:
        runner._clear_session_env(tokens)
        runner._shutdown_executor()

    assert result == {
        "platform": "telegram",
        "chat_id": "2144471399",
        "user_id": "123456",
        "session_key": "agent:main:telegram:dm:2144471399",
    }


@pytest.mark.asyncio
async def test_run_in_executor_with_context_forwards_args():
    """_run_in_executor_with_context should forward *args to the callable."""
    runner = object.__new__(GatewayRunner)

    def add(a, b):
        return a + b

    try:
        result = await runner._run_in_executor_with_context(add, 3, 7)
    finally:
        runner._shutdown_executor()
    assert result == 10


@pytest.mark.asyncio
async def test_run_in_executor_with_context_propagates_exceptions():
    """Exceptions inside the executor should propagate to the caller."""
    runner = object.__new__(GatewayRunner)

    def blow_up():
        raise ValueError("boom")

    try:
        with pytest.raises(ValueError, match="boom"):
            await runner._run_in_executor_with_context(blow_up)
    finally:
        runner._shutdown_executor()


@pytest.mark.asyncio
async def test_run_in_executor_with_context_survives_default_executor_shutdown():
    """Gateway agent work should not depend on asyncio's default executor."""
    runner = object.__new__(GatewayRunner)
    loop = asyncio.get_running_loop()

    await loop.run_in_executor(None, lambda: None)
    await loop.shutdown_default_executor()

    try:
        result = await runner._run_in_executor_with_context(lambda: "ok")
    finally:
        runner._shutdown_executor()

    assert result == "ok"


@pytest.mark.asyncio
async def test_gateway_executor_refuses_resurrection_after_shutdown():
    """A real gateway shutdown must NOT be resurrected by the recreate path.

    _shutdown_executor() means "we're stopping" — the recreate-on-shutdown
    logic exists to survive an *external* teardown of the loop default
    (test_..._survives_default_executor_shutdown), not to undo our own stop.
    """
    runner = object.__new__(GatewayRunner)

    try:
        first = await runner._run_in_executor_with_context(lambda: "first")
        assert first == "first"
        runner._shutdown_executor()

        with pytest.raises(RuntimeError, match="shutting down"):
            await runner._run_in_executor_with_context(lambda: "second")
    finally:
        runner._shutdown_executor()
