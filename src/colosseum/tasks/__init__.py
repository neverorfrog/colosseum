"""Colosseum tasks package with auto-discovery of task configurations."""

# Import training tasks to populate the colosseum task registry.
# Guarded because the deploy environment does not have mjlab.
try:
  import colosseum.tasks.dribbling  # noqa: F401
  import colosseum.tasks.velocity  # noqa: F401

  from mjlab.utils.lab_api.tasks.importer import import_packages
  _BLACKLIST_PKGS = ["utils", ".mdp"]
  import_packages(__name__, _BLACKLIST_PKGS)
except ImportError:
  pass
