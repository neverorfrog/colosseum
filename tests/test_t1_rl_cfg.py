"""Test T1 RL configuration."""

import pytest


def test_t1_rl_cfg_imports():
    """Test that T1 RL config can be imported."""
    from colosseum.tasks.velocity.rl import booster_t1_ppo_runner_cfg

    assert callable(booster_t1_ppo_runner_cfg)


def test_t1_ppo_runner_cfg_structure():
    """Test that PPO runner config returns proper structure."""
    pytest.importorskip("mjlab")  # Skip if mjlab not available

    from colosseum.tasks.velocity.rl import booster_t1_ppo_runner_cfg

    cfg = booster_t1_ppo_runner_cfg()

    # Check policy configuration
    assert cfg.policy is not None
    assert cfg.policy.init_noise_std == 1.0
    assert cfg.policy.actor_obs_normalization is True
    assert cfg.policy.critic_obs_normalization is True
    assert cfg.policy.actor_hidden_dims == (512, 256, 128)
    assert cfg.policy.critic_hidden_dims == (512, 256, 128)
    assert cfg.policy.activation == "elu"

    # Check algorithm configuration
    assert cfg.algorithm is not None
    assert cfg.algorithm.value_loss_coef == 1.0
    assert cfg.algorithm.use_clipped_value_loss is True
    assert cfg.algorithm.clip_param == 0.2
    assert cfg.algorithm.entropy_coef == 0.01
    assert cfg.algorithm.num_learning_epochs == 5
    assert cfg.algorithm.num_mini_batches == 4
    assert cfg.algorithm.learning_rate == 1.0e-3
    assert cfg.algorithm.schedule == "adaptive"
    assert cfg.algorithm.gamma == 0.99
    assert cfg.algorithm.lam == 0.95
    assert cfg.algorithm.desired_kl == 0.01
    assert cfg.algorithm.max_grad_norm == 1.0

    # Check runner configuration
    assert cfg.experiment_name == "t1_velocity"
    assert cfg.save_interval == 50
    assert cfg.num_steps_per_env == 24
    assert cfg.max_iterations == 30_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
