"""high_level_obs.py.

Class for returning high level observations to be used by task planners

Written by Will Solow & Jeff Jewett, 2026
"""

import torch


class HighLevelObs:
    """Superclass for handling high level observations."""

    def __init__(self) -> None:
        """Initialize high level observation."""

    def get_high_level_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the high level observation from the low level observation.

        The obs can be RGB images, positions, etc. High level observation used for task planning

        Args:
            obs: torch tensor of low level observations

        Returns:
            torch tensor (or other object) of high level observations

        """
        return obs
