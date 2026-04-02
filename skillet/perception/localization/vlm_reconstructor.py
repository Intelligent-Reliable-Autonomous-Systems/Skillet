import torch

from skillet.perception.localization.reconstructor_base import ReconstructorBase
from skillet.scene.base import Scene


class VLMReconstructor(ReconstructorBase):
    """Parses observations for localizing objects depth and segmentation masks."""

    def __init__(self, scene: Scene | None = None) -> None:
        """Initialize the AprilTag state estimator.

        Args:
            scene: The scene to update with the estimated poses of the AprilTags.

        """
        super().__init__(self, scene)

    def update_state(self, obs: dict[str, torch.Tensor], update: bool = True) -> None:
        """Update the state estimator with a new observation.

        Args:
            obs: The RGB-D observation to update the state estimator with.
            update: If to update the scene or not

        """
