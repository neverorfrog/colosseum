from mjlab.envs import ManagerBasedRlEnv


class ViewerCompatibleEnv(ManagerBasedRlEnv):
  """Minimal env subclass: adds get_observations() required by the mjlab viewer."""

  def get_observations(self) -> dict:
    return self.observation_manager.compute()
