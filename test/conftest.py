"""Make navigation_utils submodules importable without running its __init__ (scipy, social_path_planning)."""

import os
import sys
import types

_PKG_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "navigation_utils")
if "navigation_utils" not in sys.modules:
    _pkg = types.ModuleType("navigation_utils")
    _pkg.__path__ = [_PKG_DIR]
    sys.modules["navigation_utils"] = _pkg
