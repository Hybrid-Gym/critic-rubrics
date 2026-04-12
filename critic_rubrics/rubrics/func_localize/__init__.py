"""Function localization scoring rubric.

3-dimension evaluation of codebase navigation trajectories, transferred from
OpenHands2/evaluation/auto_prompt.
"""

from .func_localize import (
    FEATURES,
    SCORING_INSTRUCTION_MESSAGE,
    SCORING_SYSTEM_MESSAGE,
)
from .rubric_impl import FuncLocalizeRubric


TOOL_NAME = "score_trajectory"
TOOL_DESCRIPTION = "Score an agent trajectory on 3 codebase navigation dimensions (1-5 each)."

func_localize_rubrics = FuncLocalizeRubric(
    tool_name=TOOL_NAME,
    tool_description=TOOL_DESCRIPTION,
    features=FEATURES,
    system_message=SCORING_SYSTEM_MESSAGE,
    user_message=SCORING_INSTRUCTION_MESSAGE,
    rationale_description="Brief evidence (1-2 sentences) referencing specific steps in the trajectory.",
)

__all__ = [
    "FuncLocalizeRubric",
    "func_localize_rubrics",
]
