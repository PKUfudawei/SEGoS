"""Retrieval budgets are explicit and independent for the two benchmarks."""
from contextlib import contextmanager
from contextvars import ContextVar

SKILLSBENCH = dict(top_n=5, seed_top_k=4, max_skill_chars=1800, max_context_chars=9000)
ALFWORLD = dict(top_n=8, seed_top_k=5, max_skill_chars=2400, max_context_chars=12000)
_settings = ContextVar("segos_retrieval_settings", default={})


def options():
    return _settings.get()


@contextmanager
def use_settings(settings):
    settings = dict(settings or {})
    if set(settings) - set(SKILLSBENCH):
        raise ValueError("Unknown retrieval budget field")
    if any(not isinstance(v, int) or v <= 0 for v in settings.values()):
        raise ValueError("Retrieval budgets must be positive integers")
    token = _settings.set(settings)
    try:
        yield
    finally:
        _settings.reset(token)
