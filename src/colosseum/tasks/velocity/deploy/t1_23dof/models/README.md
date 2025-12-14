# Booster T1 Velocity Policy Artifacts

Place the ONNX checkpoint generated during training in this directory and name it `policy.onnx`. The deployment config references this path by default.

You can copy the exported file directly from your training run (e.g., `logs/rsl_rl/<experiment>/<run>.onnx` or the matching artifact under `wandb/run-*/files/`). Keeping the model alongside the task ensures the deployment bundle stays self-contained when syncing to the robot.
