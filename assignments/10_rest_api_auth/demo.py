"""Drives the API over real HTTP and prints a formatted report.

    python demo.py

Starts the server in a background thread, then calls every endpoint with
urllib (standard library only) so the status codes, headers and JSON bodies
shown are the real ones a client would receive.
"""
from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
import warnings

import app as api
import db

warnings.filterwarnings("ignore")

BASE = "http://127.0.0.1:5055"
GREEN, RED, YELLOW, BLUE, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[94m", "\033[1m", "\033[0m")

_server = None


def start_server():
    global _server
    db.drop_db()
    api.ensure_admin()
    _server = threading.Thread(
        target=lambda: api.create_app().run(port=5055, use_reloader=False,
                                           debug=False),
        daemon=True)
    _server.start()
    for _ in range(50):                      # wait until it answers
        try:
            call("GET", "/api/health")
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("server did not start")


def call(method: str, path: str, body: dict | None = None,
         token: str | None = None) -> tuple[int, dict]:
    """One HTTP request. Returns (status_code, decoded_json)."""
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as err:          # 4xx / 5xx are expected here
        raw = err.read().decode()
        return err.code, (json.loads(raw) if raw else {})


# ----------------------------------------------------------------- report ----
def line(label: str, method: str, path: str, status: int, note: str = "") -> None:
    colour = GREEN if status < 300 else (YELLOW if status < 500 else RED)
    print(f"    {label:<34} {method:<6} {path:<38} "
          f"{colour}{status}{RESET}  {note}")


def show(title: str, status: int, payload: dict, keys: list[str] | None = None) -> None:
    print(f"      -> {status} {json.dumps(payload)[:150]}")


def banner(text: str) -> None:
    print(f"\n{BOLD}{'=' * 78}\n  {text}\n{'=' * 78}{RESET}")


# -------------------------------------------------------------------- demo ---
def main() -> None:
    start_server()
    print(BOLD + "=" * 78)
    print("  ASSIGNMENT 10 - RESTFUL API WITH AUTHENTICATION AND AUTHORISATION")
    print("=" * 78 + RESET)
    print(f"  base url : {BASE}")
    print(f"  database : {db.DB_PATH}")

    # ---------------------------------------------------------- 1. public --
    banner("1. PUBLIC ENDPOINTS - no token required")
    s, p = call("GET", "/api/health")
    line("health check", "GET", "/api/health", s, p["status"])
    s, p = call("GET", "/api/books?per_page=3&page=1")
    line("list books (paged)", "GET", "/api/books?per_page=3", s,
         f"page {p['page']} of total {p['total']}")
    for b in p["books"]:
        print(f"         {b['id']:>2}  {b['title'][:38]:<38} {b['genre']}")
    s, p = call("GET", "/api/books?q=Martin")
    line("search 'Martin'", "GET", "/api/books?q=Martin", s,
         f"{p['count']} match(es)")
    s, p = call("GET", "/api/books?genre=database")
    line("filter genre=database", "GET", "/api/books?genre=database", s,
         f"{p['count']} match(es)")
    s, p = call("GET", "/api/books/1")
    line("get book 1", "GET", "/api/books/1", s, p["title"])
    s, p = call("GET", "/api/books/9999")
    line("get missing book", "GET", "/api/books/9999", s, p["message"])

    # -------------------------------------------------- 2. registration --
    banner("2. REGISTRATION AND LOGIN - authentication")
    s, p = call("POST", "/api/auth/register", {"username": "alice", "password": "secret123"})
    line("register alice", "POST", "/api/auth/register", s,
         f"id={p['user']['id']} role={p['user']['role']}")
    s, p = call("POST", "/api/auth/register", {"username": "bob", "password": "secret123"})
    line("register bob", "POST", "/api/auth/register", s, f"id={p['user']['id']}")
    s, p = call("POST", "/api/auth/register", {"username": "al", "password": "secret123"})
    line("username too short", "POST", "/api/auth/register", s, p["message"])
    s, p = call("POST", "/api/auth/register", {"username": "alice", "password": "secret123"})
    line("duplicate username", "POST", "/api/auth/register", s, "409 conflict")

    s, p = call("POST", "/api/auth/login", {"username": "alice", "password": "secret123"})
    user_token = p["access_token"]
    line("login alice (user)", "POST", "/api/auth/login", s,
         f"role={p['role']} expires_in={p['expires_in']}s")
    s, p = call("POST", "/api/auth/login", {"username": "admin", "password": "admin123"})
    admin_token = p["access_token"]
    line("login admin", "POST", "/api/auth/login", s, f"role={p['role']}")
    s, p = call("POST", "/api/auth/login", {"username": "alice", "password": "wrong"})
    line("login wrong password", "POST", "/api/auth/login", s, p["message"])

    print(f"\n  JWT anatomy (header.payload.signature):")
    for part, label in zip(user_token.split("."), ("header", "payload", "signature")):
        raw = part + "=" * (-len(part) % 4)
        pretty = part if label == "signature" else \
            json.dumps(json.loads(base64.urlsafe_b64decode(raw)), indent=2)
        for i, text in enumerate(str(pretty).splitlines()):
            print(f"      {label if i == 0 else '':<10} {text}")

    # ------------------------------------------------------ 3. authorise --
    banner("3. AUTHORISATION - who may do what")
    s, p = call("GET", "/api/auth/me", token=user_token)
    line("me (user token)", "GET", "/api/auth/me", s,
         f"role={p['user']['role']}")
    s, p = call("GET", "/api/auth/me")
    line("me without token", "GET", "/api/auth/me", s, "401 - no credentials")
    s, p = call("GET", "/api/auth/me", token="not.a.token")
    line("me with forged token", "GET", "/api/auth/me", s, "401 - bad signature")
    tampered = user_token[:-4] + ("aaaa" if not user_token.endswith("aaaa") else "bbbb")
    s, p = call("GET", "/api/auth/me", token=tampered)
    line("me with tampered token", "GET", "/api/auth/me", s, p["message"][:38])
    s, p = call("GET", "/api/admin/stats", token=user_token)
    line("admin area as user", "GET", "/api/admin/stats", s, "403 - forbidden")
    s, p = call("GET", "/api/admin/stats", token=admin_token)
    line("admin area as admin", "GET", "/api/admin/stats", s,
         f"{p['users']} users, {p['books']} books")

    # ---------------------------------------------------------- 4. CRUD --
    banner("4. CRUD - protected writes")
    s, p = call("POST", "/api/books", {"title": "Hacked", "author": "Nobody"})
    line("create without token", "POST", "/api/books", s, "401")
    s, p = call("POST", "/api/books", {"title": "Hacked", "author": "Nobody"},
                token=user_token)
    line("create as normal user", "POST", "/api/books", s, "403 - admin only")
    s, p = call("POST", "/api/books", {"author": "No Title"}, token=admin_token)
    line("create with bad body", "POST", "/api/books", s, p["message"])
    s, p = call("POST", "/api/books",
                {"title": "Fluent Python", "author": "Luciano Ramalho",
                 "year": 2022, "genre": "software", "copies": 3}, token=admin_token)
    new_id = p["book"]["id"]
    line("create as admin", "POST", "/api/books", s, f"id={new_id}")
    s, p = call("PUT", f"/api/books/{new_id}",
                {"copies": 9, "genre": "reference"}, token=admin_token)
    line("update as admin", "PUT", f"/api/books/{new_id}", s,
         f"copies={p['book']['copies']}")
    s, p = call("PUT", f"/api/books/{new_id}", {"author": "x"}, token=user_token)
    line("update as user", "PUT", f"/api/books/{new_id}", s, "403")
    s, p = call("DELETE", f"/api/books/{new_id}", token=user_token)
    line("delete as user", "DELETE", f"/api/books/{new_id}", s, "403")
    s, p = call("POST", "/api/auth/promote/2", token=admin_token)
    line("promote alice -> admin", "POST", "/api/auth/promote/2", s, p["message"])
    s, p = call("POST", "/api/auth/login", {"username": "alice", "password": "secret123"})
    alice_admin = p["access_token"]
    line("alice logs in again", "POST", "/api/auth/login", s, f"role={p['role']}")
    s, p = call("DELETE", f"/api/books/{new_id}", token=alice_admin)
    line("delete as promoted admin", "DELETE", f"/api/books/{new_id}", s, p["message"])
    s, p = call("DELETE", f"/api/books/{new_id}", token=admin_token)
    line("delete again", "DELETE", f"/api/books/{new_id}", s, "404 already gone")

    # ------------------------------------------------------- 5. summary --
    banner("5. FINAL STATE")
    s, p = call("GET", "/api/admin/stats", token=admin_token)
    print(f"    users            : {p['users']}")
    print(f"    books            : {p['books']}")
    print(f"    total copies     : {p['copies']}")
    print("    books by genre   :")
    for row in p["by_genre"]:
        print(f"      {row['genre']:<12} {row['n']}")
    s, p = call("GET", "/api/books?per_page=50")
    print(f"\n    {'id':>3}  {'title':<42} {'author':<26} {'yr':<5} copies")
    print("    " + "-" * 88)
    for b in p["books"]:
        print(f"    {b['id']:>3}  {b['title'][:42]:<42} {b['author'][:26]:<26} "
              f"{b['year'] or '-':<5} {b['copies']}")
    print(f"\n{RED}  Every red status above is the API correctly refusing access."
          f"{RESET}")
    print("=" * 78)


if __name__ == "__main__":
    main()
