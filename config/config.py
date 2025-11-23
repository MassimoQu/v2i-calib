"""
Compatibility wrapper for the legacy configuration helpers.

New code should prefer `calib.config.*`, but some benchmark scripts still
import `config.config`, so we forward those requests here.
"""
from configs.legacy_api import *  # noqa: F401,F403
