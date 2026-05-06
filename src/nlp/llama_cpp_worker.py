from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _write_message(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


_EMOTION_PATTERN = re.compile(r"emotion:\s*([a-zA-Z_-]+)", re.IGNORECASE)


def _messages_to_completion_prompt(messages: list[dict[str, str]]) -> str:
    emotion = ""
    dialogue: list[dict[str, str]] = []

    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system" and not emotion:
            match = _EMOTION_PATTERN.search(content)
            if match:
                emotion = match.group(1).strip().lower()
            continue
        if role in {"user", "assistant"}:
            dialogue.append({"role": role, "content": content})

    latest_user = ""
    history_items = dialogue[:-1]
    if dialogue and dialogue[-1]["role"] == "user":
        latest_user = dialogue[-1]["content"]
    elif dialogue:
        latest_user = dialogue[-1]["content"]
        history_items = dialogue[:-1]

    parts = [
        "You are a caring therapist. Reply briefly with empathy, one simple coping step, and one gentle question.",
        "User: I feel overwhelmed.",
        "Therapist: That sounds heavy. Try one slow breath. What feels hardest right now?",
        "",
    ]

    recent_history = history_items[-2:]
    for item in recent_history:
        role = "User" if item["role"] == "user" else "Therapist"
        parts.append(f"{role}: {item['content']}")

    final_user = latest_user or "I need support right now."
    if emotion:
        final_user = f"(emotion: {emotion}) {final_user}"
    parts.append(f"User: {final_user}")
    parts.append("Therapist:")
    return "\n".join(parts)


def _load_model(
    runtime_path: Path,
    model_path: Path,
    n_ctx: int,
    n_threads: int,
    n_gpu_layers: int,
):
    runtime = str(runtime_path.resolve())
    if runtime not in sys.path:
        sys.path.insert(0, runtime)

    from llama_cpp import Llama

    try:
        return Llama(
            model_path=str(model_path.resolve()),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
    except TypeError:
        return Llama(
            model_path=str(model_path.resolve()),
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )


def _generate_text(
    llm,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = _messages_to_completion_prompt(messages)
    response = llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["User:", "System:", "\n\n"],
    )
    choices = response.get("choices") or []
    text = str((choices[0] if choices else {}).get("text") or "").strip()
    if text:
        return text

    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "").strip()
            if content:
                return content
    except Exception:
        pass

    return ""


def _prime_model(llm) -> None:
    try:
        llm.create_completion(
            prompt="User: hello\nTherapist:",
            max_tokens=1,
            temperature=0.1,
            top_p=0.8,
            stop=["User:", "System:", "\n\n"],
        )
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistent llama.cpp helper worker.")
    parser.add_argument("--runtime-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    args = parser.parse_args()

    runtime_path = Path(args.runtime_path)
    model_path = Path(args.model)

    try:
        llm = _load_model(
            runtime_path=runtime_path,
            model_path=model_path,
            n_ctx=max(512, int(args.n_ctx)),
            n_threads=max(1, int(args.n_threads)),
            n_gpu_layers=int(args.n_gpu_layers),
        )
        _prime_model(llm)
    except Exception as exc:
        _write_message({"ok": False, "error": f"helper load failed: {exc}"})
        return 1

    _write_message(
        {
            "ok": True,
            "backend": "llama_cpp_helper",
            "model_path": str(model_path.resolve()),
            "runtime_path": str(runtime_path.resolve()),
        }
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except Exception as exc:
            _write_message({"ok": False, "error": f"invalid json request: {exc}"})
            continue

        command = str(request.get("command", "generate")).strip().lower()
        if command == "shutdown":
            _write_message({"ok": True})
            return 0

        if command != "generate":
            _write_message({"ok": False, "error": f"unsupported command: {command}"})
            continue

        try:
            text = _generate_text(
                llm=llm,
                messages=list(request.get("messages") or []),
                max_tokens=max(8, int(request.get("max_tokens", 24))),
                temperature=max(0.0, float(request.get("temperature", 0.6))),
                top_p=min(1.0, max(0.1, float(request.get("top_p", 0.9)))),
            )
            _write_message({"ok": True, "text": text})
        except Exception as exc:
            _write_message({"ok": False, "error": f"generation failed: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
