"""Assignment 10 - a RESTful API with a database, authentication and
authorisation.  Flask + SQLite + JWT.

    python app.py                      # start the server on :5000
    python demo.py                     # drive the API over HTTP and print a report

Resources
---------
    GET    /api/health                 public
    GET    /api/books                  public  (paged + filtered)
    GET    /api/books/<id>             public
    POST   /api/books                  admin only
    PUT    /api/books/<id>             admin only
    DELETE /api/books/<id>             admin only
    POST   /api/auth/register          public  -> creates a 'user'
    POST   /api/auth/login             public  -> returns a JWT
    GET    /api/auth/me                any logged-in user
    POST   /api/auth/promote/<user_id> admin only
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from flask import Flask, g, jsonify, request

import auth as A
import db

app = Flask(__name__)


# ------------------------------------------------------------- lifecycle ----
@app.teardown_appcontext
def close_db(exception=None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def get_conn() -> sqlite3.Connection:
    if "db" not in g:
        g.db = db.get_db()
    return g.db


# -------------------------------------------------------- error handling ----
@app.errorhandler(400)
def bad_request(err):
    return jsonify(error="bad_request", message=getattr(err, "description", "")), 400


@app.errorhandler(404)
def not_found(err):
    return jsonify(error="not_found", message="no such endpoint or record"), 404


@app.errorhandler(405)
def bad_method(err):
    return jsonify(error="method_not_allowed", message=str(err.description)), 405


@app.errorhandler(sqlite3.IntegrityError)
def conflict(err):
    return jsonify(error="conflict", message=str(err)), 409


def body() -> dict:
    """Read and validate the JSON request body."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequest("a JSON object body is required")
    return data


class BadRequest(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.description = message


@app.errorhandler(BadRequest)
def handle_bad_request(err):
    return jsonify(error="bad_request", message=err.description), 400


def row_to_dict(row) -> dict:
    return {k: row[k] for k in row.keys()}


# =========================================================== public routes ==
@app.get("/api/health")
def health():
    return jsonify(status="ok", service="library-api",
                   time=datetime.now(timezone.utc).isoformat(timespec="seconds"))


@app.get("/api/books")
def list_books():
    """Public, paginated, searchable."""
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(max(int(request.args.get("per_page", 5)), 1), 50)
    genre = request.args.get("genre")
    search = request.args.get("q")

    sql = "SELECT * FROM books WHERE 1=1"
    args: list = []
    if genre:
        sql += " AND genre = ?"
        args.append(genre)
    if search:
        sql += " AND (title LIKE ? OR author LIKE ?)"
        args += [f"%{search}%", f"%{search}%"]
    sql += " ORDER BY title LIMIT ? OFFSET ?"
    args += [per_page, (page - 1) * per_page]

    rows = get_conn().execute(sql, args).fetchall()
    total = get_conn().execute("SELECT COUNT(*) FROM books").fetchone()[0]
    return jsonify(
        page=page, per_page=per_page, total=total,
        count=len(rows), books=[row_to_dict(r) for r in rows])


@app.get("/api/books/<int:book_id>")
def get_book(book_id: int):
    row = get_conn().execute("SELECT * FROM books WHERE id = ?",
                             (book_id,)).fetchone()
    if row is None:
        return jsonify(error="not_found", message=f"no book with id {book_id}"), 404
    return jsonify(row_to_dict(row))


# =========================================================== authentication ==
@app.post("/api/auth/register")
def register():
    data = body()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if len(username) < 3:
        raise BadRequest("username must be at least 3 characters")
    if len(password) < 6:
        raise BadRequest("password must be at least 6 characters")

    conn = get_conn()
    conn.execute(
        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'user')",
        (username, A.hash_password(password)))
    conn.commit()
    user = conn.execute("SELECT id, username, role FROM users WHERE username = ?",
                        (username,)).fetchone()
    return jsonify(message="registered", user=row_to_dict(user)), 201


@app.post("/api/auth/login")
def login():
    data = body()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    user = get_conn().execute(
        "SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    # same message for a bad user and a bad password: do not leak which it was
    if user is None or not A.verify_password(password, user["password_hash"]):
        return jsonify(error="unauthorized",
                       message="invalid username or password"), 401

    token, expires_in = A.create_token(user["id"], user["username"], user["role"])
    return jsonify(message="logged in", access_token=token, token_type="Bearer",
                   expires_in=expires_in, role=user["role"])


@app.get("/api/auth/me")
@A.require_auth
def me():
    return jsonify(user=g.current_user)


@app.post("/api/auth/promote/<int:user_id>")
@A.require_role("admin")
def promote(user_id: int):
    """Only an admin may grant a role - the authorisation example."""
    conn = get_conn()
    row = conn.execute("SELECT id, username, role FROM users WHERE id = ?",
                       (user_id,)).fetchone()
    if row is None:
        return jsonify(error="not_found", message=f"no user with id {user_id}"), 404
    conn.execute("UPDATE users SET role = 'admin' WHERE id = ?", (user_id,))
    conn.commit()
    return jsonify(message="promoted", user=row_to_dict(row))


# ============================================================== write routes ==
@app.post("/api/books")
@A.require_role("admin")
def create_book():
    data = body()
    for field in ("title", "author"):
        if not (data.get(field) or "").strip():
            raise BadRequest(f"'{field}' is required")
    conn = get_conn()
    conn.execute(
        "INSERT INTO books (title, author, year, genre, copies, added_by)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (data["title"].strip(), data["author"].strip(), data.get("year"),
         data.get("genre", "unknown"), int(data.get("copies", 1)),
         g.current_user["sub"]))
    conn.commit()
    new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return jsonify(message="created", book=row_to_dict(
        conn.execute("SELECT * FROM books WHERE id = ?", (new_id,)).fetchone())), 201


@app.put("/api/books/<int:book_id>")
@A.require_role("admin")
def update_book(book_id: int):
    data = body()
    allowed = {"title", "author", "year", "genre", "copies"}
    unknown = set(data) - allowed
    if unknown:
        raise BadRequest(f"cannot update: {sorted(unknown)}")
    row = get_conn().execute("SELECT * FROM books WHERE id = ?",
                             (book_id,)).fetchone()
    if row is None:
        return jsonify(error="not_found", message=f"no book with id {book_id}"), 404
    updates = {k: v for k, v in data.items() if v is not None}
    if not updates:
        raise BadRequest("no updatable field was supplied")
    sets = ", ".join(f"{k} = ?" for k in updates)
    get_conn().execute(f"UPDATE books SET {sets} WHERE id = ?",
                       [*updates.values(), book_id])
    get_conn().commit()
    return jsonify(message="updated", book=row_to_dict(
        get_conn().execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()))


@app.delete("/api/books/<int:book_id>")
@A.require_role("admin")
def delete_book(book_id: int):
    conn = get_conn()
    cur = conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
    conn.commit()
    if cur.rowcount == 0:
        return jsonify(error="not_found", message=f"no book with id {book_id}"), 404
    return jsonify(message="deleted", id=book_id)


@app.get("/api/admin/stats")
@A.require_role("admin")
def admin_stats():
    conn = get_conn()
    return jsonify(
        users=conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
        books=conn.execute("SELECT COUNT(*) FROM books").fetchone()[0],
        copies=conn.execute("SELECT SUM(copies) FROM books").fetchone()[0],
        by_genre=[row_to_dict(r) for r in conn.execute(
            "SELECT genre, COUNT(*) AS n FROM books GROUP BY genre ORDER BY n DESC")],
    )


# ---------------------------------------------------------------- bootstrap --
def create_app(reset: bool = False):
    if reset:
        db.drop_db()
    else:
        db.init_db()
    return app


def ensure_admin(username: str = "admin", password: str = "admin123") -> None:
    """Create the first admin account if it does not exist yet."""
    conn = db.get_db()
    row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        conn.execute("INSERT INTO users (username, password_hash, role)"
                     " VALUES (?, ?, 'admin')", (username, A.hash_password(password)))
        conn.commit()
    conn.close()


if __name__ == "__main__":
    create_app()
    ensure_admin()
    print("Library API running on http://127.0.0.1:5000  (admin/admin123)")
    app.run(debug=True)
