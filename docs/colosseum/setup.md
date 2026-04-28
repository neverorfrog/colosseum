# Setup

## Requirements

- Linux x86-64
- Python 3.11
- NVIDIA GPU with CUDA 12.0+

## Installation

This project uses [Pixi](https://pixi.sh) for environment and dependency management.

```bash
git clone --recurse-submodules <repo-url>
cd colosseum
pixi install
```

`pixi install` resolves and downloads all conda and PyPI dependencies (including mjlab and colosseum itself in editable mode) into an isolated environment.

## Running scripts

Always invoke Python through pixi so the correct environment is active:

```bash
# Correct
pixi run python script.py
pixi run train task:t1-dribbling

# Wrong — will fail with ModuleNotFoundError
python script.py
```

## Verifying the installation

```bash
pixi run python -c "import colosseum; print('ok')"
```

## Environments

The workspace defines three pixi environments:

| Environment | Command | Purpose |
|-------------|---------|---------|
| `default` | `pixi run ...` | Training (requires CUDA GPU) |
| `deploy` | `pixi run -e deploy ...` | Sim-to-sim and real robot deployment |
| `docs` | `pixi run -e docs mkdocs serve` | Documentation |

## Documentation

Serve the docs site locally:

```bash
pixi run -e docs docs
```
