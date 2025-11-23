"""
Compatibility shim for the legacy ``v2x_calib`` package.

The actual implementation now lives under ``legacy.v2x_calib``; we expose it
under the original module path so existing imports keep working.
"""
import importlib
import sys

_LEGACY_MODULE = importlib.import_module('legacy.v2x_calib')

globals().update({k: v for k, v in _LEGACY_MODULE.__dict__.items() if not k.startswith('_')})
__all__ = getattr(_LEGACY_MODULE, '__all__', [k for k in globals().keys() if not k.startswith('_')])
__path__ = _LEGACY_MODULE.__path__  # type: ignore[attr-defined]
sys.modules[__name__] = _LEGACY_MODULE
