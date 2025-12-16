"""Velocity task package namespace.

Training task registration lives under `colosseum.tasks.velocity.config.*` and
should be imported explicitly by the train environment.

Deployment task registration lives under `colosseum.tasks.velocity.deploy.*` and
is auto-imported by the deploy registry. Keep this package lightweight to avoid
pulling train-only dependencies at import time.
"""

# Intentionally no side-effect imports here.
