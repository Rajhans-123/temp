"""Database layer: schema, connection handling and seed data.

SQLite is used through the standard library, so the project needs no database
server and no ORM. `get_db()` is a Flask-style per-request connection.
"""
from __future__ import annotations

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "library.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'user',   -- 'user' | 'admin'
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS books (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    title     TEXT    NOT NULL,
    author    TEXT    NOT NULL,
    year      INTEGER,
    genre     TEXT    NOT NULL DEFAULT 'unknown',
    copies    INTEGER NOT NULL DEFAULT 1,
    added_by  INTEGER REFERENCES users(id),
    created_at TEXT   NOT NULL DEFAULT (datetime('now'))
);
"""

SEED_BOOKS = [
    ("Clean Code",              "Robert C. Martin",      2008, "software",   4),
    ("Designing Data-Intensive Applications", "Martin Kleppmann", 2017, "software", 3),
    ("The Pragmatic Programmer", "Andrew Hunt",         1999, "software",   2),
    ("Database System Concepts", "Abraham Silberschatz", 2020, "database",  5),
    ("Mining of Massive Datasets", "Jure Leskovec",      2014, "database",   3),
    ("Refactoring",             "Martin Fowler",         2018, "software",   2),
]


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row           # rows behave like dicts
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(seed: bool = True) -> None:
    conn = get_db()
    conn.executescript(SCHEMA)
    if seed and conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 0:
        conn.executemany(
            "INSERT INTO books (title, author, year, genre, copies)"
            " VALUES (?, ?, ?, ?, ?)", SEED_BOOKS)
        conn.commit()
    conn.close()


def drop_db() -> None:
    """Used by the demo so it always starts from a known state."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db()
