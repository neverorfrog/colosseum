try:
  import colosseum.tasks.velocity.config.t1_23dof.t1_velocity_cfg  # noqa: F401
  import colosseum.tasks.velocity.config.t1_23dof_amp.t1_velocity_cfg  # noqa: F401
  import colosseum.tasks.velocity.config.t1_23dof_manu.t1_velocity_manu_cfg  # noqa: F401
  import colosseum.tasks.velocity.config.t1_23dof_rma.t1_velocity_rma_cfg  # noqa: F401
  import colosseum.tasks.velocity.config.t1_12dof.t1_velocity_cfg  # noqa: F401
  import colosseum.tasks.velocity.config.t1_booster.t1_velocity_cfg  # noqa: F401
except ImportError:
  pass
