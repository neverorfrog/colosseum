from colosseum.config.types.logger import LoggerConfig

WANDB = LoggerConfig(
  enabled=True,
  project="colosseum",
  mode="online",
  log_interval=50,
  save_interval=100_000,
  log_dir="./logs",
  console_level="INFO",
)

DISABLED = LoggerConfig(
  enabled=False,
  mode="disabled",
)

DEFAULTS = {
  "wandb": WANDB,
  "disabled": DISABLED,
}
