"""
Compatibility shim for legacy visualization utilities now located in
``legacy.visualize``.
"""
import importlib
import sys

_LEGACY_VIS = importlib.import_module('legacy.visualize')
globals().update({k: v for k, v in _LEGACY_VIS.__dict__.items() if not k.startswith('_')})
__all__ = getattr(_LEGACY_VIS, '__all__', [k for k in globals().keys() if not k.startswith('_')])
__path__ = _LEGACY_VIS.__path__  # type: ignore[attr-defined]
sys.modules[__name__] = _LEGACY_VIS
