"""Authentication and authorisation.

Authentication = "who are you?"   -> verified by a signed JWT token
Authorisation  = "what may you do?" -> checked by the @require_auth / @require_role
                                      decorators

Both are done with the standard library only:
* passwords  -> PBKDF2-HMAC-SHA256 with a random salt
* tokens     -> JWT (HS256) built with hmac + base64url
"""
from __future__ import annotations

import base64
import binascii
import functools
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Callable

# In production this comes from the environment, never from the source file.
SECRET_KEY = os.environ.get("API_SECRET_KEY", "dev-secret-change-me")
TOKEN_TTL_SECONDS = 3600
PBKDF2_ROUNDS = 120_000
ROLES = ("user", "admin")


# ------------------------------------------------------------- passwords ----
def hash_password(password: str) -> str:
    """pbkdf2$<rounds>$<salt_hex>$<hash_hex>"""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)
    return f"pbkdf2${PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, rounds, salt_hex, hash_hex = stored.split("$")
        expected = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                       bytes.fromhex(salt_hex), int(rounds))
    except (ValueError, AttributeError):
        return False
    # constant-time compare so the check cannot be timed
    return hmac.compare_digest(expected.hex(), hash_hex)


# ------------------------------------------------------------------ JWT -----
def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def create_token(user_id: int, username: str, role: str) -> tuple[str, int]:
    """Return (token, expires_in_seconds)."""
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
        "jti": secrets.token_hex(8),          # unique id, so tokens can be revoked
    }
    signing_input = (f"{_b64(json.dumps(header).encode())}."
                     f"{_b64(json.dumps(payload).encode())}")
    signature = hmac.new(SECRET_KEY.encode(), signing_input.encode(),
                         hashlib.sha256).digest()
    return f"{signing_input}.{_b64(signature)}", TOKEN_TTL_SECONDS


class TokenError(Exception):
    """Raised when a token is missing, malformed, forged or expired."""


def decode_token(token: str) -> dict:
    try:
        signing_input, signature = token.rsplit(".", 1)
        header_b64, payload_b64 = signing_input.split(".", 1)
        given = _unb64(signature)
    except (ValueError, TypeError, binascii.Error, AttributeError):
        raise TokenError("token is not a valid JWT")
    try:
        expected = hmac.new(SECRET_KEY.encode(), signing_input.encode(),
                            hashlib.sha256).digest()
    except (UnicodeError, AttributeError):
        raise TokenError("token could not be verified")
    if not hmac.compare_digest(given, expected):
        raise TokenError("signature does not match - the token was tampered with")
    try:
        payload = json.loads(_unb64(payload_b64))
    except (ValueError, binascii.Error):
        raise TokenError("token payload is not valid JSON")
    if payload.get("exp", 0) < time.time():
        raise TokenError("token has expired")
    return payload


# ------------------------------------------------------------- decorators ---
def require_auth(func: Callable) -> Callable:
    """401 unless a valid Bearer token is present. Sets g.current_user."""
    from flask import g, jsonify, request

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify(error="unauthorized",
                           message="send 'Authorization: Bearer <token>'"), 401
        try:
            payload = decode_token(header[7:].strip())
        except TokenError as err:
            return jsonify(error="unauthorized", message=str(err)), 401
        g.current_user = payload
        return func(*args, **kwargs)
    return wrapper


def require_role(*roles: str) -> Callable:
    """403 unless the authenticated user's role is in `roles`."""
    from flask import g, jsonify

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        @require_auth
        def wrapper(*args, **kwargs):
            if g.current_user["role"] not in roles:
                return jsonify(
                    error="forbidden",
                    message=f"role '{g.current_user['role']}' may not do this, "
                            f"needs {list(roles)}"), 403
            return func(*args, **kwargs)
        return wrapper
    return decorator
