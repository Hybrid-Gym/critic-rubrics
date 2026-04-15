"""Function localization rubric: scoring rubric for code navigation trajectories.

Evaluates agent trajectories on 5 dimensions across three levels:
  - Stage-level: workflow_adherence
  - Strategy-level: file_localization, function_localization, effective_file_editing
  - Action-level: tool_call_correctness
"""

from typing import Literal

from ...feature import Feature
from ...prediction import ClassificationPrediction


ScorePrediction = ClassificationPrediction[Literal["1", "2", "3", "4", "5"]]

SCORING_SYSTEM_MESSAGE = """You are an expert evaluator of AI coding agent trajectories. You will analyze an agent's behavior across 5 dimensions at three evaluation levels (stage, strategy, action).

========================
SCORING PHILOSOPHY
========================
For each dimension, score from 1 (worst) to 5 (best) based on how well the agent's
behavior matches the desirable qualities and avoids the undesirable ones described below.
Use your judgment — no fixed rubric table is provided for each score level.

========================
QUALITY STANDARDS
========================
- Evidence-based: Reference specific agent actions (step numbers, commands, files).
- Use the full 1-5 range as you see fit based on the criteria described.
"""

SCORING_INSTRUCTION_MESSAGE = """=== END OF AGENT TRAJECTORY ===

Score the agent's behavior by calling the score_trajectory function.

For each of the 5 dimensions, provide:
1) A score from "1" to "5" (as a string)
2) A brief rationale referencing specific steps

─────────────────────────────────────────
Dimension 1: workflow_adherence (stage-level)
─────────────────────────────────────────
Does the agent follow a logical progression: locate candidate file(s) → locate code
lines in the file(s) → understand the code?

Desirable: clear, ordered stages where each builds on the previous; no stage is skipped.
Undesirable: jumping around without progression, skipping stages, or repeating stages unnecessarily.

─────────────────────────────────────────
Dimension 2: file_localization (strategy-level)
─────────────────────────────────────────
How effectively does the agent locate the relevant file(s)?

Desirable: uses targeted keyword searches, tries alternative terms when initial searches fail,
narrows candidate files to a small set before reading them.
Undesirable: guesses file paths without searching, uses only one keyword and never iterates,
proceeds with a large candidate set.

─────────────────────────────────────────
Dimension 3: function_localization (strategy-level)
─────────────────────────────────────────
How effectively does the agent pinpoint the exact target function/class within files?

Desirable: uses keyword search to find the exact line of the target, reads only a focused
region around it, confirms the match (e.g., checking the signature).
Undesirable: reads entire files blindly without prior search, reads large chunks without
narrowing to the relevant section.

─────────────────────────────────────────
Dimension 4: effective_file_editing (strategy-level)
─────────────────────────────────────────
How well does the agent approach the editing process?

Desirable: reads the target code before editing, reasons about what to change, makes
precise and correct edits, verifies the result by re-reading the modified section.
Undesirable: edits without reading the code first, makes imprecise or incorrect changes,
skips verification after editing.

─────────────────────────────────────────
Dimension 5: tool_call_correctness (action-level)
─────────────────────────────────────────
Does the agent use file-edit and other tools correctly without errors?

Desirable: tool calls succeed on the first attempt with correct arguments (file paths,
line ranges, syntax).
Undesirable: tool calls produce errors (wrong arguments, file not found, syntax mistakes),
repeated failed attempts without diagnosing the cause.
"""

FEATURES = [
    Feature(
        name="workflow_adherence",
        description=(
            "Stage-level: Does the agent follow locate file(s) → locate code lines → "
            "understand code in a clear, ordered progression?"
        ),
        prediction_type=ScorePrediction,
    ),
    Feature(
        name="file_localization",
        description=(
            "Strategy-level: Does the agent use targeted keyword searches, try alternative terms "
            "on failure, and narrow candidates to a small set before reading?"
        ),
        prediction_type=ScorePrediction,
    ),
    Feature(
        name="function_localization",
        description=(
            "Strategy-level: Does the agent pinpoint the exact target function/class via "
            "keyword search and read only the relevant region?"
        ),
        prediction_type=ScorePrediction,
    ),
    Feature(
        name="effective_file_editing",
        description=(
            "Strategy-level: Does the agent read code before editing, make precise changes, "
            "and verify the result?"
        ),
        prediction_type=ScorePrediction,
    ),
    Feature(
        name="tool_call_correctness",
        description=(
            "Action-level: Do file-edit and other tool calls succeed without errors "
            "(correct arguments, no syntax mistakes, no repeated failures)?"
        ),
        prediction_type=ScorePrediction,
    ),
]
