# Dribbling — Research Code

Paper-specific code for the dribbling publication (visual RMA with depth encoder).

## Structure

```
research/dribbling/
└── scripts/
    ├── pipeline_dribbling.py   # Full curriculum pipeline (Phase 1 + Phase 2 per stage)
    ├── train_phase2.py         # Phase 2 visual adaptation encoder training
    └── evaluate_dribbling.py   # Evaluation protocol
```

Python modules specific to this paper live in `src/colosseum/research/dribbling/`:
- `encoders.py` — `DepthEncoder`, `BallHead`, `ObstacleHead`
- `rma_terms.py` — `DribblingRmaTerm` and `DribblingRmaTermCfg`

When these modules stabilise they will be promoted into core colosseum.

## Running

```bash
# Full curriculum (Phase 1 + Phase 2 per stage)
pixi run -e train pipeline-dribbling

# Phase 2 only (from a Phase 1 checkpoint)
pixi run -e train train-phase2 --checkpoint ./logs/run/checkpoints/latest.pt task:t1-dribbling

# Evaluation
pixi run -e train eval-dribbling task:t1-dribbling --checkpoint ./logs/run/checkpoints/latest.pt
```
