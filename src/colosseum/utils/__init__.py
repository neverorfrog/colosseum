import os
from pathlib import Path

def project_root() -> Path:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    max_iterations = 100  # Set a limit for the number of iterations
    for _ in range(max_iterations):
        if (
            "requirements.txt" in os.listdir(current_dir)
            or "setup.py" in os.listdir(current_dir)
            or "pyproject.toml" in os.listdir(current_dir)
        ):
            return Path(current_dir)
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError(
        "requirements.txt not found in any parent directories within the iteration limit"
    )

def src_dir() -> Path:
    return Path(project_root()) / "src" / "colosseum"