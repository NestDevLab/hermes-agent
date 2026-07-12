"""Fail-closed persistence tests for resolver-produced gateway context."""

from copy import deepcopy
import json
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, normalize_agent_memory_context


def _tag(namespace: str, digit: str) -> str:
    return f"hmac-sha256:{namespace}:{digit * 64}"


def _group_source(user_id: str = "external-a") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="room-1",
        chat_type="group",
        user_id=user_id,
        user_name="Untrusted display name",
    )


def _group_context() -> dict:
    room_tag = _tag("room", "1")
    person_a_tag = _tag("person", "2")
    person_b_tag = _tag("person", "3")
    relationship_tag = _tag("relationship", "4")
    return {
        "schema": "amf-conversation-context/v1",
        "conversation_kind": "group",
        "actor_id": "agent:character",
        "participant_identity_ids": ["person:a", "person:b"],
        "person_bindings": {
            "external-a": "person:a",
            "external-b": "person:b",
        },
        "allowed_scopes": [
            {"type": "agent", "id": "agent:character"},
            {"type": "person", "id": "person:a"},
            {"type": "relationship", "id": "relationship:a:b"},
            {"type": "room", "id": "room:one"},
        ],
        "scope_ids": [
            "agent:character",
            "person:a",
            "relationship:a:b",
            "room:one",
        ],
        "room_scope_id": "room:one",
        "context_tags": {
            "room": [room_tag],
            "person": [person_a_tag, person_b_tag],
            "relationship": [relationship_tag],
        },
        "context_tag_bindings": {
            "room": {"room:one": [room_tag]},
            "person": {
                "person:a": [person_a_tag],
                "person:b": [person_b_tag],
            },
            "relationship": {"relationship:a:b": [relationship_tag]},
        },
        "output_policy": {
            "schema": "amf-group-output-policy/v1",
            "policy_id": "policy-room-one-v1",
            "allowed_claim_ids": ["mem_safe_claim"],
            "protected_claim_ids": ["mem_private_claim"],
            "protected_subject_ids": ["person:b"],
        },
    }


def _store(tmp_path, config=None) -> SessionStore:
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config or GatewayConfig())
    store._db = None
    store._loaded = False
    return store


def test_valid_context_is_canonical_and_roundtrips(tmp_path):
    source = _group_source()
    context = _group_context()
    store = _store(tmp_path)
    entry = store.get_or_create_session(source)

    assert store.bind_agent_memory_context(entry.session_key, source, context) is True

    persisted = _store(tmp_path).get_or_create_session(source)
    assert persisted.agent_memory_context == context


def test_missing_current_sender_binding_clears_previous_context(tmp_path):
    source = _group_source()
    store = _store(tmp_path)
    entry = store.get_or_create_session(source)
    assert store.bind_agent_memory_context(entry.session_key, source, _group_context())

    invalid = deepcopy(_group_context())
    del invalid["person_bindings"][source.user_id]
    assert store.bind_agent_memory_context(entry.session_key, source, invalid) is False
    assert entry.agent_memory_context is None

    persisted = _store(tmp_path).get_or_create_session(source)
    assert persisted.agent_memory_context is None


def test_route_mismatch_cannot_bind_context(tmp_path):
    source = _group_source()
    other_route = SessionSource(
        platform=Platform.DISCORD,
        chat_id="room-2",
        chat_type="group",
        user_id="external-a",
    )
    store = _store(tmp_path)
    entry = store.get_or_create_session(source)

    assert store.bind_agent_memory_context(entry.session_key, other_route, _group_context()) is False
    assert entry.agent_memory_context is None


def test_additional_or_ambiguous_metadata_is_rejected():
    source = _group_source()
    context = _group_context()
    context["participant_display_names"] = ["do not trust this"]

    assert normalize_agent_memory_context(context, source) is None


def test_group_context_requires_structured_output_policy():
    source = _group_source()
    context = _group_context()
    context["output_policy"] = None

    assert normalize_agent_memory_context(context, source) is None


def test_sender_primary_and_alt_aliases_may_resolve_to_the_same_identity():
    source = _group_source()
    source.user_id_alt = "external-a-alt"
    context = _group_context()
    context["person_bindings"] = {
        "external-a": "person:a",
        "external-a-alt": "person:a",
        "external-b": "person:b",
    }

    assert normalize_agent_memory_context(context, source) is not None


def test_shared_session_persists_sender_b_context_when_origin_is_sender_a(tmp_path):
    config = GatewayConfig(group_sessions_per_user=False)
    source_a = _group_source("external-a")
    source_b = _group_source("external-b")
    store = _store(tmp_path, config)
    entry_a = store.get_or_create_session(source_a)
    entry_b = store.get_or_create_session(source_b)
    assert entry_b is entry_a
    assert entry_b.origin.user_id == "external-a"

    context_b = _group_context()
    context_b["person_bindings"] = {"external-b": "person:b"}
    context_b["allowed_scopes"] = [
        {"type": "agent", "id": "agent:character"},
        {"type": "person", "id": "person:b"},
        {"type": "relationship", "id": "relationship:a:b"},
        {"type": "room", "id": "room:one"},
    ]
    context_b["scope_ids"] = [
        "agent:character", "person:b", "relationship:a:b", "room:one"
    ]
    assert store.bind_agent_memory_context(entry_b.session_key, source_b, context_b)

    serialized = entry_b.to_dict()
    assert serialized["agent_memory_context"] == context_b
    restarted = _store(tmp_path, config).get_or_create_session(source_a)
    assert restarted.agent_memory_context == context_b


def test_reload_still_drops_structurally_invalid_context(tmp_path):
    source = _group_source()
    store = _store(tmp_path)
    entry = store.get_or_create_session(source)
    assert store.bind_agent_memory_context(entry.session_key, source, _group_context())

    sessions_file = tmp_path / "sessions.json"
    data = json.loads(sessions_file.read_text(encoding="utf8"))
    data[entry.session_key]["agent_memory_context"]["scope_ids"] = ["room:wrong"]
    sessions_file.write_text(json.dumps(data), encoding="utf8")

    restarted = _store(tmp_path).get_or_create_session(source)
    assert restarted.agent_memory_context is None
