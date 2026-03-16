"""Colosseum tasks package with auto-discovery of task configurations."""

# Import tasks to populate the colosseum task registry
import colosseum.tasks.velocity  # noqa: F401

try:
    from mjlab.utils.lab_api.tasks.importer import import_packages
    _BLACKLIST_PKGS = ["utils", ".mdp"]
    import_packages(__name__, _BLACKLIST_PKGS)
except ImportError:
    pass
