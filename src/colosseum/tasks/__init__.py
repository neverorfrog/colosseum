"""Colosseum tasks package with auto-discovery of task configurations."""

import importlib
import pkgutil

_BLACKLIST = {"utils", "mdp"}

for _info in pkgutil.iter_modules(__path__, __name__ + "."):
  _short = _info.name.rsplit(".", 1)[-1]
  if _short in _BLACKLIST:
    continue
  try:
    importlib.import_module(_info.name)
  except ImportError:
    pass
