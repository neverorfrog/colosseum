try:
  import colosseum.tasks.locomotion.config.t1.t1_locomotion_cfg  # noqa: F401
  import colosseum.tasks.locomotion.config.t1_rma.t1_locomotion_rma_cfg  # noqa: F401
  import colosseum.tasks.locomotion.config.t1_amp.t1_locomotion_amp_cfg  # noqa: F401
except ImportError:
  print("ERROR IN IMPORTING TASK")
