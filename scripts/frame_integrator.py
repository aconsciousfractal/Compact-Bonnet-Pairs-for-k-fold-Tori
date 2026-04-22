"""Compatibility re-export for the paper-cited `scripts/` path.

Canonical implementation lives in `src.frame_integrator`.
"""

from src import frame_integrator as _impl

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
