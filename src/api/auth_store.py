from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path("data/processed")
USERS_PATH = DATA_DIR / "users.json"
SESSIONS_PATH = DATA_DIR / "sessions.json"
PROFILE_PHOTO_DIR = Path("data/outputs/profile_photos")


def _ensure_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_PATH.exists():
        USERS_PATH.write_text("[]", encoding="utf-8")
    if not SESSIONS_PATH.exists():
        SESSIONS_PATH.write_text("{}", encoding="utf-8")


def _read_json(path: Path, fallback: Any) -> Any:
    _ensure_files()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _write_json(path: Path, data: Any) -> None:
    _ensure_files()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    )
    return dk.hex()


def create_user(
    email: str,
    username: str,
    sex: str,
    age: int,
    password: str,
    photo_path: str | None = None,
) -> dict:
    users = _read_json(USERS_PATH, [])
    if any(u["email"].lower() == email.lower() for u in users):
        raise ValueError("Email deja utilise.")
    if any(u["username"].lower() == username.lower() for u in users):
        raise ValueError("Username deja utilise.")

    salt = secrets.token_hex(16)
    user = {
        "id": secrets.token_hex(12),
        "email": email,
        "username": username,
        "sex": sex,
        "age": age,
        "password_salt": salt,
        "password_hash": _hash_password(password, salt),
        "photo_path": photo_path,
        # Legacy local store now skips token-based email verification.
        "is_verified": True,
        "created_at": _now_iso(),
    }
    users.append(user)
    _write_json(USERS_PATH, users)
    return user

def authenticate(email: str, password: str) -> dict:
    users = _read_json(USERS_PATH, [])
    for user in users:
        if user["email"].lower() != email.lower():
            continue
        expected = _hash_password(password, user["password_salt"])
        if not hmac.compare_digest(expected, user["password_hash"]):
            break
        if not user.get("is_verified", False):
            raise ValueError("Email non verifie.")
        return user
    raise ValueError("Identifiants invalides.")


def create_session(user_id: str) -> str:
    sessions = _read_json(SESSIONS_PATH, {})
    token = secrets.token_urlsafe(32)
    sessions[token] = {"user_id": user_id, "created_at": _now_iso()}
    _write_json(SESSIONS_PATH, sessions)
    return token


def get_user_by_session(token: str) -> dict:
    sessions = _read_json(SESSIONS_PATH, {})
    session = sessions.get(token)
    if not session:
        raise ValueError("Session invalide.")

    users = _read_json(USERS_PATH, [])
    user_id = session["user_id"]
    for user in users:
        if user["id"] == user_id:
            return user
    raise ValueError("Utilisateur introuvable.")


def public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "email": user["email"],
        "username": user["username"],
        "sex": user["sex"],
        "age": user["age"],
        "photo_path": user.get("photo_path"),
        "is_verified": user.get("is_verified", False),
        "created_at": user.get("created_at"),
    }
