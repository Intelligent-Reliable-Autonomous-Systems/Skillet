from typing import Literal

from conditional_repair.orcam.orcam import (
    ORCAMHypothesisSet,
    UPDomainConverter,
    learn_orcam_step,
    step_orcam,
)
from unified_planning.model import Object as UPObject
from unified_planning.model import UPState
from unified_planning.plans import ActionInstance

from skillet.planning import AbstractModel
from skillet.planning.abstract.up_utils import AbstractAction, up_state_to_dict


class ORCAMLearningAgent:
    """An ORCAM learning agent."""

    def __init__(self) -> None:
        """Construct the ORCAM agent."""
        self._orcam: ORCAMHypothesisSet | None = None

    def initialize(self, abstract_model: AbstractModel) -> None:
        """Initialize the ORCAM agent."""
        problem = abstract_model.problem
        converter = UPDomainConverter(problem)
        self._orcam = ORCAMHypothesisSet(converter.actions, converter.predicates, converter)

    def sample_action(self, up_state: UPState, up_objects: list[UPObject]) -> tuple[AbstractAction, ActionInstance]:
        """ORCAM active learning."""
        if self._orcam is None:
            raise ValueError("ORCAM agent not initialized")
        dict_state = up_state_to_dict(up_state)
        up_action = step_orcam(self._orcam, dict_state, up_objects)
        abstract_action = AbstractAction(
            action=up_action.action.name, parameters=[p.object().name for p in up_action.actual_parameters]
        )
        return abstract_action, up_action

    def update(
        self,
        up_state: UPState,
        up_objects: list[UPObject],
        up_action: ActionInstance,
        next_up_state: UPState,
        execution: Literal["applicable", "inapplicable"],
    ) -> None:
        """Update the ORCAM agent."""
        if self._orcam is None:
            raise ValueError("ORCAM agent not initialized")
        dict_state = up_state_to_dict(up_state)
        dict_next_state = up_state_to_dict(next_up_state)
        learn_orcam_step(self._orcam, dict_state, up_objects, up_action, dict_next_state, execution)
