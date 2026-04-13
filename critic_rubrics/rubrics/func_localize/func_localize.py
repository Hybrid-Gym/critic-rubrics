"""Function localization rubric: scoring rubric for code navigation trajectories.

Transferred from OpenHands2/evaluation/auto_prompt. Evaluates agent trajectories
on three dimensions of codebase navigation: structure discovery, file localization,
and function localization.
"""

from typing import Literal

from ...feature import Feature
from ...prediction import ClassificationPrediction


ScorePrediction = ClassificationPrediction[Literal["1", "2", "3", "4", "5"]]

SCORING_SYSTEM_MESSAGE = """You are an expert evaluator of AI coding agent trajectories. You will analyze an agent's behavior across 3 dimensions of codebase navigation quality.

========================
SCORING SCALE
========================
For each rule, assign a score from 1 to 5:
- 1: Major violation — agent completely ignored this dimension
- 2: Significant gap — agent skipped important steps or fell into repeated loops
- 3: Adequate — agent followed the general flow but with notable deviations
- 4: Good — agent performed well with only minor deviations
- 5: Excellent — agent closely followed best practices for this dimension

========================
QUALITY STANDARDS
========================
- Evidence-based: Reference specific agent actions (step numbers, commands, files).
- Conservative: When evidence is ambiguous, lean toward the middle score (3).
"""

SCORING_INSTRUCTION_MESSAGE = """=== END OF AGENT TRAJECTORY ===

Score the agent's behavior by calling the score_trajectory function.

For each of the 3 dimensions, provide:
1) A score from "1" to "5" (as a string)
2) A brief rationale referencing specific steps

Dimensions to evaluate:
1. structure_discovery — Did the agent explore the repo's directory structure (e.g., ls, tree) before searching? Merely counting files does not count.
2. file_localization — Did the agent use targeted keywords to find the right file, switching strategy when searches failed? It should keep searching for new keywords until the set of candidate files is limited.
3. function_localization — Did the agent narrow down to the exact target within candidate files? It should use keyword searching to determine the exact line where the function is located.
"""

FEATURES = [
    Feature(
        name="structure_discovery",
        description=(
            "Did the agent explore the repo's directory structure (e.g., ls, tree, find with directory listing) "
            "before searching for specific code? Merely counting files (find|wc -l) without viewing the layout "
            "does not count. Score 1-5."
        ),
        prediction_type=ScorePrediction,
    ),
    Feature(
        name="file_localization",
        description=(
            "Did the agent use targeted keywords to find the right file, switching strategy when initial searches failed? "
            "It should keep searching for new keywords until the set of candidate files is limited to a small number. Score 1-5."
        ),
        prediction_type=ScorePrediction,
    ),
    Feature(
        name="function_localization",
        description=(
            "Did the agent narrow down to the exact target function/class within candidate files? "
            "It should use keyword searching to determine the exact line where the function is located. Score 1-5."
        ),
        prediction_type=ScorePrediction,
    ),
]
