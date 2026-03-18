"""Generate a minimal identity ONNX model for testing Policy inference.

The model has:
  input:  float32[1, OBS_DIM]  named "obs"
  output: float32[1, OBS_DIM]  named "action"
  op:     Identity (output = input, no parameters)

Usage:
    pixi run -e train python tools/make_identity_onnx.py [obs_dim] [out_path]

Defaults: obs_dim=82 (T1 23-DOF velocity task), out_path=models/identity.onnx
"""

import sys
import os
import numpy as np
import onnx
from onnx import helper, TensorProto

def make_identity_model(obs_dim: int) -> onnx.ModelProto:
    obs    = helper.make_tensor_value_info("obs",    TensorProto.FLOAT, [1, obs_dim])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, obs_dim])
    node   = helper.make_node("Identity", inputs=["obs"], outputs=["action"])
    graph  = helper.make_graph([node], "identity", [obs], [action])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

if __name__ == "__main__":
    obs_dim  = int(sys.argv[1])      if len(sys.argv) > 1 else 82
    out_path = sys.argv[2]           if len(sys.argv) > 2 else "models/identity.onnx"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model = make_identity_model(obs_dim)
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"Saved identity model ({obs_dim}→{obs_dim}) to {out_path}")
