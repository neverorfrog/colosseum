# Colosseum

Learning playground for humanoid robots related to robot soccer


## Troubleshooting

### Viser viewer: `MjlabViserScene` cannot be instantiated

When launching with `--viewer viser`, you may see:

```
TypeError: Can't instantiate abstract class MjlabViserScene with abstract method add_rectangle
```

This is a known bug in the installed mjlab: `MjlabViserScene` inherits from `DebugVisualizer` which declares `add_rectangle` as abstract, but the Viser backend does not implement it yet.

**Fix:** add the missing no-op stub to the installed mjlab. Open `.pixi/envs/train/lib/python3.11/site-packages/mjlab/viewer/viser/scene.py` and add the following method to `MjlabViserScene`, just before `clear()` (around line 427):

```python
@override
def add_rectangle(self, *args, **kwargs) -> None:  # type: ignore[override]
    pass
```