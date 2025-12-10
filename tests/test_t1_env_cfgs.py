"""Test T1 environment configurations."""

import pytest


def test_t1_env_cfgs_imports():
  """Test that T1 env_cfgs can be imported."""
  # This will fail if there are syntax errors or missing imports
  from colosseum.tasks.velocity.config.t1 import (
    booster_t1_flat_env_cfg,
    booster_t1_rough_env_cfg,
  )

  assert callable(booster_t1_flat_env_cfg)
  assert callable(booster_t1_rough_env_cfg)


def test_t1_rough_env_cfg_structure():
  """Test that rough env cfg returns proper structure."""
  pytest.importorskip("mujoco")  # Skip if mujoco not available

  from colosseum.tasks.velocity.config.t1 import booster_t1_rough_env_cfg

  cfg = booster_t1_rough_env_cfg(play=False)

  # Check scene configuration
  assert "robot" in cfg.scene.entities
  assert len(cfg.scene.sensors) == 2  # feet_ground_contact and self_collision

  # Check viewer
  assert cfg.viewer.body_name == "Trunk"

  # Check actions configured
  assert "joint_pos" in cfg.actions

  # Check rewards configured
  assert "pose" in cfg.rewards
  assert "upright" in cfg.rewards
  assert "body_ang_vel" in cfg.rewards
  assert "self_collisions" in cfg.rewards

  # Check observations configured
  assert "policy" in cfg.observations
  assert "critic" in cfg.observations


def test_t1_flat_env_cfg_structure():
  """Test that flat env cfg returns proper structure."""
  pytest.importorskip("mujoco")  # Skip if mujoco not available

  from colosseum.tasks.velocity.config.t1 import booster_t1_flat_env_cfg

  cfg = booster_t1_flat_env_cfg(play=False)

  # Check terrain is flat
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert cfg.scene.terrain.terrain_generator is None


def test_t1_play_mode():
  """Test that play mode applies correct overrides."""
  pytest.importorskip("mujoco")  # Skip if mujoco not available

  from colosseum.tasks.velocity.config.t1 import booster_t1_rough_env_cfg

  cfg = booster_t1_rough_env_cfg(play=True)

  # Check episode length is very long
  assert cfg.episode_length_s == int(1e9)

  # Check corruption is disabled
  assert cfg.observations["policy"].enable_corruption is False

  # Check push event is removed
  assert "push_robot" not in cfg.events


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
