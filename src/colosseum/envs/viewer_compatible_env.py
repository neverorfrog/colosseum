from mjlab.envs import ManagerBasedRlEnv


class ViewerCompatibleEnv(ManagerBasedRlEnv):
  """Minimal env subclass:
  - adds get_observations() required by the mjlab viewer
  - registers viz_callbacks from the env config into manager_visualizers

  Any subclass config that declares
    viz_callbacks: list[tuple[str, Callable[[env], object_with_debug_vis]]]
  will have those callbacks registered automatically.
  """

  def get_observations(self) -> dict:
    return self.observation_manager.compute()

  def setup_manager_visualizers(self) -> None:
    super().setup_manager_visualizers()
    for name, factory in getattr(self.cfg, "viz_callbacks", []):
      self.manager_visualizers[name] = factory(self)
