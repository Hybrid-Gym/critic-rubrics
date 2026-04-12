#!/usr/bin/env python3
"""Test the func_localize rubric on a single trajectory from the data."""

import gzip
import json
import sys
from pathlib import Path


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from critic_rubrics import Annotator
from critic_rubrics.rubrics.func_localize import func_localize_rubrics


DATA_PATH = "/home/gaokaizhang/integration/func_localize_claude_success1438.jsonl.gz"
CONFIG_PATH = Path("/home/gaokaizhang/integration/config.toml")


def load_model_config(name: str) -> dict[str, str]:
    """Load a model's config from config.toml."""
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)
    entry = cfg["llm"][name]
    return {"model": entry["model"], "api_key": entry["api_key"]}


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt4o-mini"
    try:
        model_cfg = load_model_config(model_name)
    except KeyError:
        with open(CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
        print(f"Unknown model: {model_name}. Choose from: {list(cfg['llm'].keys())}")
        return 1

    print(f"Using model: {model_cfg['model']}")

    # Load first trajectory
    with gzip.open(DATA_PATH, "rt") as f:
        data = json.loads(f.readline())

    instance_id = data["instance_id"]
    print(f"Instance: {instance_id}")
    print(f"History entries: {len(data.get('history', []))}")

    rubric = func_localize_rubrics
    request = rubric.create_annotation_request(data)
    if request is None:
        print("ERROR: create_annotation_request returned None")
        return 1

    print(f"Features: {len(rubric.features)}")
    print()

    print(f"Sending scoring request to {model_cfg['model']}...")
    response = Annotator.annotate(
        request,
        model=model_cfg["model"],
        api_key=model_cfg["api_key"],
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        print("ERROR: No tool calls in response")
        print("Response:", response.choices[0].message.content)
        return 1

    tool_call = tool_calls[0].model_dump()
    features = rubric.tool_call_to_feature_data(tool_call)

    print(f"Extracted {len(features)} features:")
    for fd in features:
        pred = fd.prediction.to_dict()
        pred.pop("type")
        print(f"  {fd.feature.name}: score={pred.get('label', '?')}")
        print(f"    rationale: {pred.get('rationale', '')[:120]}")

    return 0


if __name__ == "__main__":
    exit(main())
