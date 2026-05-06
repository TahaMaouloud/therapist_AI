from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.nlp import therapist_agent as agent


def test_engine_falls_back_to_helper_when_inline_load_fails() -> None:
    engine = agent._LocalLlamaEngine()

    with (
        patch.object(engine, "_provider", return_value="llama_cpp"),
        patch.object(engine, "_model_path", return_value=Path("LLMA")),
        patch.object(engine, "_prefer_llama_helper", return_value=False),
        patch.object(engine, "_load_llama_cpp", side_effect=RuntimeError("inline failed")),
        patch.object(engine, "_load_llama_cpp_helper", return_value=None),
    ):
        assert engine._ensure_loaded() is True

    assert engine.backend_name() == "llama_cpp_helper"


def test_backend_status_reports_helper_python_path() -> None:
    engine = agent._LLM_ENGINE
    original_loaded = engine._loaded
    original_loading_started = engine._loading_started
    original_backend = engine._backend
    original_failure_reason = engine._failure_reason
    original_runtime_path = engine._resolved_runtime_path
    original_helper_python_path = engine._resolved_helper_python_path

    try:
        engine._loaded = True
        engine._loading_started = False
        engine._backend = "llama_cpp_helper"
        engine._failure_reason = ""
        engine._resolved_runtime_path = str(Path("vendor/llama_cpp_runtime312"))
        engine._resolved_helper_python_path = str(Path(".tmp/python312_runtime/python.exe"))

        status = agent.therapist_backend_status()

        assert status["backend"] == "llama_cpp_helper"
        assert status["runtime_path"].endswith("vendor\\llama_cpp_runtime312")
        assert status["helper_python_path"].endswith(".tmp\\python312_runtime\\python.exe")
    finally:
        engine._loaded = original_loaded
        engine._loading_started = original_loading_started
        engine._backend = original_backend
        engine._failure_reason = original_failure_reason
        engine._resolved_runtime_path = original_runtime_path
        engine._resolved_helper_python_path = original_helper_python_path
