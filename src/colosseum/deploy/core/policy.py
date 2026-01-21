"""Policy base class for Colosseum deployment controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from colosseum.deploy.core.base_controller import BaseController

class Policy(ABC):
    """Base class for all deployment policies.

    All policy implementations must:
    1. Inherit from this class
    2. Implement __init__(self, controller: BaseController)
    3. Implement reset() and inference() methods

    The constructor takes a BaseController instance which provides access to:
    - controller.cfg: Complete ControllerConfig (including PolicyConfig)
    - controller.robot: Robot state and commands
    - controller.vel_command: Velocity commands (if applicable)
    """

    def __init__(self, controller: "BaseController"):
        """Initialize policy from controller.

        Args:
            controller: BaseController instance providing access to config,
                       robot state, and commands.
        """
        # Access everything from controller
        self.controller = controller
        self.config = controller.cfg.policy  # PolicyConfig
        self.robot = controller.robot

        # Load trained model artifact (ONNX, TorchScript, or checkpoint)
        model_path = Path(self.config.checkpoint_path)
        if not model_path.is_absolute():
            # Relative to this file
            model_path = Path(__file__).parent / model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"[Policy] Loading model from: {model_path}")
        self._model: _PolicyModule = self._load_artifact(model_path)
        self._model.eval()

        # Action scaling (must match training!)
        # Training uses: joint_target = action * scale + default_joint_pos
        # where scale is just ACTION_SCALE (uniform 0.25 for all joints)
        # NO effort_limit/stiffness calculation in training!
        action_scale = self.config.action_scale
        if isinstance(action_scale, dict):
            # Convert dict to tensor in simulation joint order
            action_scale_list = [action_scale.get(name, 0.25) for name in self.robot.cfg.sim_joint_names]
            self.action_scale = torch.tensor(action_scale_list, dtype=torch.float32)
        elif isinstance(action_scale, (int, float)):
            # Uniform scaling for all joints (default case)
            self.action_scale = torch.full((self.robot.num_joints,), action_scale, dtype=torch.float32)
        else:
            self.action_scale = torch.tensor(action_scale, dtype=torch.float32)

    @abstractmethod
    def reset(self) -> None:
        """Reset policy state.

        Called when the controller starts, before the first inference step.
        """
        pass

    @abstractmethod
    def inference(self) -> torch.Tensor:
        """Run policy inference for one step.

        Returns:
            Action tensor for this step (joint position targets).
        """
        raise NotImplementedError

    def _load_artifact(self, model_path: Path) -> "_PolicyModule":
        """Resolve a model artifact based on file extension."""
        try:
            import onnxruntime as ort
        except ImportError as err:  # pragma: no cover - env issue
            raise RuntimeError(
                "onnxruntime is required to load ONNX checkpoints. Install it "
                "via `pip install onnxruntime`."
            ) from err

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else [
            "CPUExecutionProvider"
        ]
        session = ort.InferenceSession(str(model_path), providers=providers)

        return _OnnxPolicyWrapper(session)

@runtime_checkable
class _PolicyModule(Protocol):
    def __call__(self, obs: torch.Tensor) -> torch.Tensor: ...

    def eval(self) -> "_PolicyModule": ...


class _OnnxPolicyWrapper:
    """Lightweight callable that mirrors ``torch.nn.Module`` for ONNXRuntime."""

    def __init__(self, session) -> None:  # type: ignore[no-untyped-def]
        self.session = session
        self._input_names = [inp.name for inp in self.session.get_inputs()]
        self._output_names = [out.name for out in self.session.get_outputs()]

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple bridge
        ort_inputs = {name: obs.detach().cpu().numpy() for name in self._input_names}
        outputs = self.session.run(self._output_names, ort_inputs)
        result = torch.from_numpy(outputs[0])
        return result.to(obs.device)

    def eval(self) -> "_OnnxPolicyWrapper":
        return self