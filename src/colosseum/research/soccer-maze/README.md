# Soccer Maze — Research Code

Paper-specific code for the soccer-maze publication.

## Structure

```
research/soccer-maze/
└── scripts/        # Evaluation and ablation scripts (to be added)
```

The soccer-maze task itself lives in `src/colosseum/tasks/soccer_maze/`.
Paper-specific Python modules (if any) will go in `src/colosseum/research/soccer_maze/`.

## Running

```bash
# Training
pixi run -e train train task:t1-soccer-maze
```
