"""Convert OpenHands trajectory history into messages for the func_localize annotator.

Transforms the raw history (action/observation pairs) into a compact step-by-step
representation, then wraps it with the task description for the LLM judge.
"""

import logging
from typing import Any

from litellm import AllMessageValues as LiteLLMMessageType, ChatCompletionSystemMessage, ChatCompletionUserMessage


logger = logging.getLogger(__name__)


def extract_steps(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract agent action steps with their observations from OpenHands history."""
    steps = []
    for i, entry in enumerate(history):
        if entry.get("source") != "agent":
            continue
        if "args" not in entry or "observation" in entry:
            continue
        atype = entry.get("action", "")
        if atype in ("system", "recall", "message"):
            continue
        obs = None
        for j in range(i + 1, min(i + 3, len(history))):
            if "observation" in history[j]:
                obs = history[j]
                break
        steps.append({"action_type": atype, "args": entry.get("args", {}), "observation": obs})
    return steps


def format_action_short(action_type: str, args: dict[str, Any]) -> str:
    """Format an action into a short human-readable string."""
    if action_type == "run":
        cmd = args.get("command", "")[:120]
        return f'run("{cmd}")'
    elif action_type == "read":
        path = args.get("path", "")
        s, e = args.get("start"), args.get("end")
        return f'read("{path}", {s}-{e})' if s and e else f'read("{path}")'
    elif action_type == "edit":
        path = args.get("path", "")
        old = str(args.get("old_str", ""))[:60].replace("\n", "\u21b5")
        return f'edit("{path}", old="{old}...")'
    elif action_type == "think":
        t = args.get("thought", "")[:80].replace("\n", " ")
        return f'think("{t}...")'
    elif action_type == "finish":
        return "finish()"
    elif action_type == "condensation":
        return "condensation()"
    return f"{action_type}(...)"


def format_observation(obs: dict[str, Any] | None, max_chars: int = 300) -> str:
    """Format an observation dict into a short string."""
    if obs is None:
        return "(no response)"
    content = obs.get("content", "") or obs.get("message", "")
    content = str(content).strip()
    if not content:
        return "(empty response)"
    if len(content) > max_chars:
        content = content[:max_chars] + f"... [{len(content)} chars total]"
    return content


def format_trajectory(history: list[dict[str, Any]], max_steps: int = 60) -> str:
    """Format a trajectory into a readable step-by-step action log."""
    steps = extract_steps(history)
    lines = []
    for i, step in enumerate(steps[:max_steps]):
        action_str = format_action_short(step["action_type"], step["args"])
        obs_str = format_observation(step.get("observation"), max_chars=300)
        lines.append(f"Step {i + 1} [{step['action_type']}]: {action_str}")
        lines.append(f"  -> {obs_str}")
    if len(steps) > max_steps:
        lines.append(f"... ({len(steps) - max_steps} more steps omitted)")
    return "\n".join(lines)


def get_task_description(data: dict[str, Any]) -> str:
    """Extract the task description from a trajectory's first user message."""
    for e in data.get("history") or []:
        if e.get("source") == "user" and e.get("action") == "message":
            return e.get("args", {}).get("content", "")[:600]
    # Fallback to instruction field
    return str(data.get("instruction", ""))[:600]


def transform_for_annotator(
    data: dict[str, Any],
    system_message: str,
    annotation_instruction_message: str,
    max_steps: int = 60,
) -> list[LiteLLMMessageType] | None:
    """Transform OpenHands trajectory data into messages for the scoring LLM.

    Args:
        data: A single trajectory record with 'history' and optionally 'instruction'.
        system_message: The evaluator system prompt.
        annotation_instruction_message: Instructions appended after the trajectory.
        max_steps: Maximum steps to include from the trajectory.

    Returns:
        List of messages for the LLM, or None if the trajectory is empty.
    """
    history = data.get("history") or []
    if not history:
        logger.warning("Empty history, skipping.")
        return None

    task_desc = get_task_description(data)
    if not task_desc:
        logger.warning("No task description found, skipping.")
        return None

    trajectory_text = format_trajectory(history, max_steps=max_steps)
    if not trajectory_text.strip():
        logger.warning("Empty trajectory after formatting, skipping.")
        return None

    user_content = (
        f"## Task Description\n{task_desc}\n\n"
        f"## Agent Trajectory\n{trajectory_text}\n\n"
        f"{annotation_instruction_message}"
    )

    return [
        ChatCompletionSystemMessage(role="system", content=system_message),
        ChatCompletionUserMessage(role="user", content=user_content),
    ]
