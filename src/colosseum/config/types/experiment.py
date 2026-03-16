from __future__ import annotations

import dataclasses
from pathlib import Path

import tyro
import yaml
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated

import colosseum.config.values.algorithm
import colosseum.config.values.logger
import colosseum.config.values.task
from colosseum.config.types.algorithm import AlgorithmConfig
from colosseum.config.types.logger import LoggerConfig
from colosseum.config.types.task import TaskConfig, get_task_class


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class BaseExperimentConfig:
    """Base configuration shared between training and playing."""

    use_cuda: bool = True
    checkpoint: str | None = None

    task: Annotated[
        TaskConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(
                colosseum.config.values.task.DEFAULTS
            )
        ),
    ] = dataclasses.field(default_factory=lambda: list(colosseum.config.values.task.DEFAULTS.values())[0])

    algo: Annotated[
        AlgorithmConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(
                colosseum.config.values.algorithm.DEFAULTS
            )
        ),
    ] = dataclasses.field(default_factory=lambda: colosseum.config.values.algorithm.PPO_DEFAULT)


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class TrainConfig(BaseExperimentConfig):
    """Training-specific configuration."""

    name: str = "EXPERIMENT"
    seed: int = 42

    logger: Annotated[
        LoggerConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(
                colosseum.config.values.logger.DEFAULTS
            )
        ),
    ] = dataclasses.field(default_factory=lambda: colosseum.config.values.logger.WANDB)

    def save_config(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(
                self.to_serializable_dict(), f, default_flow_style=False, sort_keys=False
            )

    def to_serializable_dict(self) -> dict:
        from enum import Enum
        from typing import cast

        def is_serializable(obj):
            if obj is None:
                return True
            if isinstance(obj, (str, int, float, bool)):
                return True
            if isinstance(obj, Enum):
                return True
            if isinstance(obj, (list, tuple, dict)):
                return True
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return True
            return False

        def convert(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                result = {}
                for k, v in obj.__dict__.items():
                    if is_serializable(v):
                        result[k] = convert(v)
                    elif isinstance(v, (list, tuple)):
                        converted_items = [
                            convert(item) for item in v if is_serializable(item)
                        ]
                        if converted_items:
                            result[k] = converted_items
                return result
            elif isinstance(obj, (list, tuple)):
                return type(obj)(convert(item) for item in obj if is_serializable(item))
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items() if is_serializable(v)}
            else:
                return obj

        return cast(dict, convert(self))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if "task" in data:
            task_data = data["task"]
            task_name = task_data["name"]
            TaskClass = get_task_class(task_name)
            data["task"] = TaskClass.reconstruct_from_dict(task_data)

        if "algo" in data:
            from colosseum.config.types.algorithm import get_algorithm_config_class
            algo_data = data["algo"]
            algo_name = algo_data["name"]
            AlgoConfigClass = get_algorithm_config_class(algo_name)
            data["algo"] = AlgoConfigClass.reconstruct_from_dict(algo_data)

        if "logger" in data:
            data["logger"] = LoggerConfig(**data["logger"])

        return cls(**data)


ExperimentConfig = TrainConfig
