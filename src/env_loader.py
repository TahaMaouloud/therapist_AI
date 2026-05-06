from __future__ import annotations

import os
import threading
from pathlib import Path

_ENV_LOCK = threading.Lock()
_ENV_LOADED = False


def load_local_env_file(path: str | Path = ".env") -> None:
    """Load a local .env file once without overriding existing environment variables."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    with _ENV_LOCK:
        if _ENV_LOADED:
            return

        env_path = Path(path)
        if not env_path.exists():
            _ENV_LOADED = True
            return

        try:
            lines = env_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            _ENV_LOADED = True
            return

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            if line.lower().startswith("export "):
                line = line[7:].strip()
                if "=" not in line:
                    continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue

            value = value.strip()
            if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value

        _ENV_LOADED = True
