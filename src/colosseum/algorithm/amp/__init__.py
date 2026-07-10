"""AMP (Adversarial Motion Priors) components for residual kick training."""

from colosseum.algorithm.amp.discriminator import AMPDiscriminator
from colosseum.algorithm.amp.motion_dataset import AmpMotionDataset
from colosseum.algorithm.amp.replay_buffer import AMPReplayBuffer

__all__ = ["AMPDiscriminator", "AmpMotionDataset", "AMPReplayBuffer"]
