#!/usr/bin/env python3
"""Run both rubric pipelines on a subset of func_localize data.

Tests:
1. The repo's existing trajectory annotation rubric (via raw_completions)
2. Our new func_localize scoring rubric (via history)

Usage:
    python scripts/run_subset_pipeline.py [--n 3] [--model gpt4o-mini]
"""

import argparse
import gzip
import json
import time
from pathlib import Path


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from critic_rubrics import Annotator
from critic_rubrics.rubrics.func_localize import func_localize_rubrics
from critic_rubrics.rubrics.trajectory import annotate_conversation_rubrics


DATA_PATH = Path("/home/gaokaizhang/integration/func_localize_claude_success1438.jsonl.gz")
CONFIG_PATH = Path("/home/gaokaizhang/integration/config.toml")
OUTPUT_DIR = Path("/home/gaokaizhang/integration/critic-rubrics/output")


def load_model_config(name: str) -> dict[str, str]:
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)
    entry = cfg["llm"][name]
    return {"model": entry["model"], "api_key": entry["api_key"]}


def load_subset(n: int) -> list[dict]:
    records = []
    with gzip.open(DATA_PATH, "rt") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            records.append(json.loads(line))
    return records


def _strip_temperature_for_o_series(request: dict, model: str) -> None:
    """O-series models (o3, o1) only support temperature=1; drop it."""
    model_lower = model.lower()
    if ("o3" in model_lower or "o1" in model_lower) and "temperature" in request:
        request.pop("temperature")


def run_trajectory_rubric(record: dict, model: str, api_key: str) -> dict | None:
    """Run the repo's existing trajectory annotation rubric."""
    inputs = record.get("raw_completions")
    if not inputs:
        return None

    rubric = annotate_conversation_rubrics
    request = rubric.create_annotation_request(inputs)
    if request is None:
        return None

    _strip_temperature_for_o_series(request, model)
    response = Annotator.annotate(request, model=model, api_key=api_key)
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        return None

    tool_call = tool_calls[0].model_dump()
    features = rubric.tool_call_to_feature_data(tool_call)
    return {fd.feature.name: fd.prediction.to_dict() for fd in features}


def run_func_localize_rubric(record: dict, model: str, api_key: str) -> dict | None:
    """Run our new func_localize scoring rubric."""
    rubric = func_localize_rubrics
    request = rubric.create_annotation_request(record)
    if request is None:
        return None

    response = Annotator.annotate(request, model=model, api_key=api_key)
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        return None

    tool_call = tool_calls[0].model_dump()
    features = rubric.tool_call_to_feature_data(tool_call)
    return {fd.feature.name: fd.prediction.to_dict() for fd in features}


def main():
    parser = argparse.ArgumentParser(description="Run both rubric pipelines on a data subset")
    parser.add_argument("--n", type=int, default=3, help="Number of records to process")
    parser.add_argument("--model", type=str, default="gpt4o-mini", help="Model name from config.toml")
    args = parser.parse_args()

    model_cfg = load_model_config(args.model)
    model, api_key = model_cfg["model"], model_cfg["api_key"]
    print(f"Model: {model}")
    print(f"Processing {args.n} records\n")

    records = load_subset(args.n)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pipeline 1: Existing trajectory annotation rubric ──
    print("=" * 60)
    print("PIPELINE 1: Trajectory Annotation (repo's existing rubric)")
    print("=" * 60)
    traj_results = []
    for i, rec in enumerate(records):
        iid = rec["instance_id"]
        print(f"  [{i+1}/{len(records)}] {iid}...", end=" ", flush=True)
        t0 = time.time()
        try:
            features = run_trajectory_rubric(rec, model, api_key)
            elapsed = time.time() - t0
            if features:
                traj_results.append({"instance_id": iid, "features": features})
                n_feat = len(features)
                print(f"OK ({n_feat} features, {elapsed:.1f}s)")
            else:
                traj_results.append({"instance_id": iid, "error": "no features extracted"})
                print(f"SKIP (no features, {elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            traj_results.append({"instance_id": iid, "error": str(e)})
            print(f"ERROR: {e} ({elapsed:.1f}s)")

    traj_out = OUTPUT_DIR / "trajectory_annotations.jsonl"
    with open(traj_out, "w") as f:
        for r in traj_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved to {traj_out}")

    # Print summary
    ok = [r for r in traj_results if "features" in r]
    print(f"Success: {len(ok)}/{len(traj_results)}")
    if ok:
        sample = ok[0]["features"]
        print(f"Sample features ({ok[0]['instance_id']}):")
        for name, pred in list(sample.items())[:5]:
            pred_copy = dict(pred)
            ptype = pred_copy.pop("type", "?")
            val = pred_copy.get("label") or pred_copy.get("detected") or pred_copy.get("text", "")
            if isinstance(val, str) and len(val) > 80:
                val = val[:80] + "..."
            print(f"    {name} ({ptype}): {val}")
        if len(sample) > 5:
            print(f"    ... and {len(sample) - 5} more")

    # ── Pipeline 2: Func localize scoring rubric ──
    print()
    print("=" * 60)
    print("PIPELINE 2: Func Localize Scoring (our new rubric)")
    print("=" * 60)
    fl_results = []
    for i, rec in enumerate(records):
        iid = rec["instance_id"]
        print(f"  [{i+1}/{len(records)}] {iid}...", end=" ", flush=True)
        t0 = time.time()
        try:
            features = run_func_localize_rubric(rec, model, api_key)
            elapsed = time.time() - t0
            if features:
                fl_results.append({"instance_id": iid, "features": features})
                scores = {k: v.get("label") for k, v in features.items() if "label" in v}
                print(f"OK (scores: {scores}, {elapsed:.1f}s)")
            else:
                fl_results.append({"instance_id": iid, "error": "no features extracted"})
                print(f"SKIP (no features, {elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            fl_results.append({"instance_id": iid, "error": str(e)})
            print(f"ERROR: {e} ({elapsed:.1f}s)")

    fl_out = OUTPUT_DIR / "func_localize_scores.jsonl"
    with open(fl_out, "w") as f:
        for r in fl_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved to {fl_out}")

    # Print score summary
    ok = [r for r in fl_results if "features" in r]
    print(f"Success: {len(ok)}/{len(fl_results)}")
    if ok:
        print("\nScore summary across instances:")
        rule_names = ["workflow_adherence", "file_localization", "function_localization", "effective_file_editing", "tool_call_correctness"]
        for rule in rule_names:
            vals = []
            for r in ok:
                pred = r["features"].get(rule)
                if pred and "label" in pred:
                    vals.append(int(pred["label"]))
            if vals:
                avg = sum(vals) / len(vals)
                dist = " ".join(str(v) for v in vals)
                print(f"  {rule}: avg={avg:.2f}  [{dist}]")


if __name__ == "__main__":
    main()
