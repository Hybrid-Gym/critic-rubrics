"""Rubric implementation for func_localize scoring."""

from typing import Any

from litellm import ChatCompletionRequest

from ..base import BaseRubrics
from .converter import transform_for_annotator


class FuncLocalizeRubric(BaseRubrics):
    """Rubric for scoring function-localization agent trajectories on 3 navigation dimensions."""

    max_steps: int = 60

    def create_annotation_request(
        self,
        inputs: dict[str, Any],
        model: str = "openai/o3-2025-04-16",
    ) -> ChatCompletionRequest | None:
        """Create an annotation request from an OpenHands trajectory record.

        Args:
            inputs: A trajectory record with 'history' (and optionally 'instruction').
            model: LLM model to use for scoring.

        Returns:
            ChatCompletionRequest or None if the trajectory can't be processed.
        """
        assert self.user_message is not None, "user_message (scoring instruction) must be defined"
        messages = transform_for_annotator(
            inputs,
            system_message=self.system_message,
            annotation_instruction_message=self.user_message,
            max_steps=self.max_steps,
        )
        if messages is None:
            return None
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            tools=self.tools,
            tool_choice=self.tool_choice,
        )
        # O-series models (o3, o3-mini) only support temperature=1;
        # skip temperature for them, set 0.0 for others.
        model_lower = model.lower()
        if "o3" not in model_lower and "o1" not in model_lower:
            request["temperature"] = 0.0
        return request
