"""Custom exception classes and the "exception cleaning" layer.

Exception cleaning means three things, and this module does all three:

1. TRANSLATE - a low level error (FileNotFoundError, ValueError, sqlite3...)
   is wrapped in a meaningful application error that carries a machine readable
   code, so callers never have to guess what went wrong.
2. CLEAN UP  - if an operation half-finished (a temp file, a locked resource)
   it is removed / released before the error propagates.
3. PROTECT   - the user sees a short friendly message; the full traceback stays
   in the log file, never on the console.
"""
from __future__ import annotations

import functools
import os
import tempfile
from typing import Callable


# --------------------------------------------------------------- exceptions --
class AppError(Exception):
    """Base class for every error this application raises on purpose."""

    code = "APP_ERROR"
    status_hint = "check the input you supplied"

    def __init__(self, message: str, *, details: dict | None = None,
                 cause: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def as_dict(self) -> dict:
        return {"error": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ValidationError(AppError):
    code = "VALIDATION_ERROR"
    status_hint = "one or more fields are invalid"


class StorageError(AppError):
    code = "STORAGE_ERROR"
    status_hint = "the file could not be read or written"


class ProcessingError(AppError):
    code = "PROCESSING_ERROR"
    status_hint = "a work item failed while being processed"


# ------------------------------------------------------------------ helpers --
def _remove_quietly(path: str | None) -> bool:
    """Delete a file without ever raising - cleanup must not mask the real error."""
    if not path or not os.path.exists(path):
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False


class TempFile:
    """Context manager that gives a temp file and removes it, even on error.

    This is the 'clean up after a partial failure' half of exception cleaning.
    """

    def __init__(self, suffix: str = ".tmp"):
        self.suffix = suffix
        self.path: str | None = None
        self._removed = False

    def __enter__(self) -> str:
        fd, self.path = tempfile.mkstemp(suffix=self.suffix)
        os.close(fd)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._removed = _remove_quietly(self.path)
        return False           # never swallow the exception


def translate(exc: Exception, *, during: str) -> AppError:
    """Map any built-in exception onto the right AppError subclass."""
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, FileNotFoundError):
        return StorageError(f"{during}: file not found", cause=exc)
    if isinstance(exc, (PermissionError, IsADirectoryError, OSError)):
        return StorageError(f"{during}: the operating system refused access", cause=exc)
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return ValidationError(f"{during}: {exc}", cause=exc)
    return ProcessingError(f"{during}: unexpected {type(exc).__name__}", cause=exc)


def clean_call(fn: Callable, *args, during: str = "operation", **kwargs):
    """Run fn(), translating and cleaning any exception it raises."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:                       # noqa: BLE001 - on purpose
        raise translate(exc, during=during) from exc


def clean_method(logger, *, during: str, log=None):
    """Decorator: clean + translate + log a method's exceptions.

        @clean_method(logger, during="saving record")
        def save(self, record): ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:                # noqa: BLE001
                app_error = translate(exc, during=during)
                (log or logger.error)(
                    "%s failed: %s", func.__name__, app_error,
                    exc_info=app_error.cause or exc,   # traceback -> log file only
                )
                raise app_error from app_error.cause
        return wrapper
    return decorator
