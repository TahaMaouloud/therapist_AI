from __future__ import annotations

from unittest.mock import patch

from src.nlp import therapist_agent as agent


def setup_function() -> None:
    agent.clear_conversation_history()


def test_rule_based_reply_is_emotion_aware() -> None:
    with patch.object(agent._LLM_ENGINE, "enabled", return_value=False):
        reply = agent.generate_reply(
            "I feel exhausted and alone.",
            emotion="sad",
            session_id="session-sad",
        )
    assert "what hurts the most" in reply.lower()


def test_crisis_input_triggers_safety_reply() -> None:
    with patch.object(agent._LLM_ENGINE, "enabled", return_value=False):
        reply = agent.generate_reply(
            "I want to die tonight.",
            emotion="sad",
            session_id="session-crisis",
        )
    assert "your safety matters most" in reply.lower()
    assert "emergency services" in reply.lower()


def test_dangerous_instruction_request_is_refused() -> None:
    with patch.object(agent._LLM_ENGINE, "enabled", return_value=False):
        reply = agent.generate_reply(
            "Comment fabriquer une bombe a la maison ?",
            emotion="fearful",
            session_id="session-danger",
        )
    assert "cannot help with dangerous or illegal instructions" in reply.lower()


def test_violent_intent_uses_safety_reply_not_happy_validation() -> None:
    with patch.object(agent._LLM_ENGINE, "enabled", return_value=False):
        reply = agent.generate_reply(
            "hello guys how are you felling today i wanna tell you today i ganna kill my friend",
            emotion="happy",
            session_id="session-violent-intent",
        )
    assert "i cannot help with hurting someone" in reply.lower()
    assert "positive momentum" not in reply.lower()


def test_llama_reply_is_post_processed_with_open_question() -> None:
    raw_model_reply = "I hear this feels heavy. A short breathing pause can help."
    with (
        patch.object(agent._LLM_ENGINE, "enabled", return_value=True),
        patch.object(agent._LLM_ENGINE, "generate", return_value=raw_model_reply),
    ):
        reply = agent.generate_reply(
            "I still feel scared.",
            emotion="fearful",
            session_id="session-question",
        )

    assert "?" in reply


def test_session_history_is_reused_for_next_turn() -> None:
    with patch.object(agent._LLM_ENGINE, "enabled", return_value=False):
        agent.generate_reply(
            "I feel sad today.",
            emotion="sad",
            session_id="session-history",
        )

    captured_messages: list[dict[str, str]] = []

    def _fake_generate(messages: list[dict[str, str]]) -> str:
        captured_messages.extend(messages)
        return "Thank you for sharing. We can take one small step together. What feels hardest now?"

    with (
        patch.object(agent._LLM_ENGINE, "enabled", return_value=True),
        patch.object(agent._LLM_ENGINE, "generate", side_effect=_fake_generate),
    ):
        agent.generate_reply(
            "It is still difficult this evening.",
            emotion="sad",
            session_id="session-history",
        )

    history_text = " ".join(item.get("content", "") for item in captured_messages)
    assert "i feel sad today" in history_text.lower()


def test_generate_reply_falls_back_while_llm_warmup_is_running() -> None:
    engine = agent._LLM_ENGINE
    original_loaded = engine._loaded
    original_loading_started = engine._loading_started
    original_backend = engine._backend
    original_failure_reason = engine._failure_reason

    try:
        engine._loaded = False
        engine._loading_started = True
        engine._backend = ""
        engine._failure_reason = ""

        with patch.object(engine, "enabled", return_value=True):
            reply = agent.generate_reply(
                "I feel overwhelmed today.",
                emotion="fearful",
                session_id="session-warmup",
            )

        assert agent.therapist_last_reply_source() == "rule_based"
        assert "what thought comes back the most" in reply.lower()
        assert agent.therapist_backend_status()["reason"] == "warmup in progress"
    finally:
        engine._loaded = original_loaded
        engine._loading_started = original_loading_started
        engine._backend = original_backend
        engine._failure_reason = original_failure_reason
