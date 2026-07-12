"""Gateway-side collection rules for trusted routing-context resolvers."""

from types import SimpleNamespace

from gateway.run import GatewayRunner


def _resolve(monkeypatch, results):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: results)
    runner = object.__new__(GatewayRunner)
    return runner._resolve_gateway_routing_context(
        event=object(),
        source=object(),
        session_entry=SimpleNamespace(session_key="agent:main:test:dm:one"),
    )


def test_exactly_one_namespaced_candidate_is_accepted(monkeypatch):
    context = {"schema": "amf-conversation-context/v1"}
    assert _resolve(monkeypatch, [{"agent_memory_context": context}]) is context


def test_missing_or_multiple_candidates_fail_closed(monkeypatch):
    context = {"schema": "amf-conversation-context/v1"}
    assert _resolve(monkeypatch, []) is None
    assert _resolve(
        monkeypatch,
        [
            {"agent_memory_context": context},
            {"agent_memory_context": dict(context)},
        ],
    ) is None


def test_extra_resolver_fields_fail_closed(monkeypatch):
    assert _resolve(
        monkeypatch,
        [{"agent_memory_context": {}, "untrusted_hint": "ignored"}],
    ) is None
