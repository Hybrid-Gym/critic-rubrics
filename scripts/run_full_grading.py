#!/usr/bin/env python3
"""Grade all trajectories in func_localize_claude_success1438.jsonl.gz using func_localize rubrics.

For each trajectory, scores 5 dimensions (workflow_adherence, file_localization,
function_localization, effective_file_editing, tool_call_correctness) on 1-5 scale,
then computes the average.

Writes per-record results to output JSONL and maintains run.md with live progress.
Uses ThreadPoolExecutor for parallel API calls.
"""

import gzip
import json
import sys
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from critic_rubrics import Annotator
from critic_rubrics.rubrics.func_localize import func_localize_rubrics

DATA_PATH = Path("/home/gaokaizhang/integration/func_localize_claude_success1438.jsonl.gz")
CONFIG_PATH = Path("/home/gaokaizhang/integration/config.toml")
OUTPUT_DIR = Path("/home/gaokaizhang/integration/critic-rubrics/output")
RUN_MD = Path("/home/gaokaizhang/integration/critic-rubrics/run.md")

DIMS = ["workflow_adherence", "file_localization", "function_localization", "effective_file_editing", "tool_call_correctness"]


def load_model_config(name: str) -> dict[str, str]:
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)
    entry = cfg["llm"][name]
    return {"model": entry["model"], "api_key": entry["api_key"]}


def update_run_md(
    total: int,
    done: int,
    ok: int,
    errors: int,
    skipped: int,
    dim_scores: dict[str, list[int]],
    combined_avgs: list[float],
    model: str,
    start_time: float,
    workers: int = 1,
    finished: bool = False,
):
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0

    lines = [
        f"# Grading Run: func_localize rubrics",
        f"",
        f"- **Model**: `{model}`",
        f"- **Data**: `{DATA_PATH.name}` ({total} trajectories)",
        f"- **Workers**: {workers}",
        f"- **Status**: {'COMPLETED' if finished else 'RUNNING'}",
        f"- **Progress**: {done}/{total} ({done*100/total:.1f}%)",
        f"- **Success/Error/Skip**: {ok}/{errors}/{skipped}",
        f"- **Elapsed**: {elapsed/60:.1f} min | Rate: {rate:.2f} traj/s",
    ]
    if not finished:
        lines.append(f"- **ETA**: ~{eta/60:.1f} min remaining")
    lines.append("")
    lines.append("## Average Scores (across successful)")
    for dim in DIMS:
        vals = dim_scores.get(dim, [])
        if vals:
            avg = sum(vals) / len(vals)
            lines.append(f"- **{dim}**: {avg:.3f} (n={len(vals)})")
        else:
            lines.append(f"- **{dim}**: n/a")
    if combined_avgs:
        overall = sum(combined_avgs) / len(combined_avgs)
        lines.append(f"- **average_of_5**: {overall:.3f} (n={len(combined_avgs)})")
    else:
        lines.append(f"- **average_of_5**: n/a")
    lines.append("")

    RUN_MD.write_text("\n".join(lines))


def score_one(record: dict, rubric, model: str, api_key: str) -> dict:
    """Score a single trajectory. Returns a result dict."""
    iid = record["instance_id"]
    t0 = time.time()
    try:
        request = rubric.create_annotation_request(record)
        if request is None:
            return {"instance_id": iid, "skip": "create_annotation_request returned None", "_elapsed": time.time() - t0}

        response = Annotator.annotate(request, model=model, api_key=api_key)
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            return {"instance_id": iid, "error": "no tool calls in response", "_elapsed": time.time() - t0}

        tool_call = tool_calls[0].model_dump()
        features = rubric.tool_call_to_feature_data(tool_call)
        scores = {}
        feature_details = {}
        for fd in features:
            pred = fd.prediction.to_dict()
            pred.pop("type", None)
            feature_details[fd.feature.name] = pred
            if "label" in pred:
                scores[fd.feature.name] = int(pred["label"])

        avg = sum(scores.values()) / len(scores) if scores else 0
        return {
            "instance_id": iid,
            "scores": scores,
            "average": round(avg, 4),
            "details": feature_details,
            "_elapsed": time.time() - t0,
        }
    except Exception as e:
        return {
            "instance_id": iid,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "_elapsed": time.time() - t0,
        }


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt5-mini"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    model_cfg = load_model_config(model_name)
    model, api_key = model_cfg["model"], model_cfg["api_key"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"func_localize_scores_{model_name}_full.jsonl"

    # Load all records
    records = []
    with gzip.open(DATA_PATH, "rt") as f:
        for line in f:
            records.append(json.loads(line))
    total = len(records)

    # Resume support
    already_done = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                rec = json.loads(line)
                already_done.add(rec["instance_id"])
        print(f"Resuming: {len(already_done)} already completed")

    print(f"Model: {model}")
    print(f"Workers: {workers}")
    print(f"Total trajectories: {total}")
    print(f"Remaining: {total - len(already_done)}")
    print(f"Output: {output_file}")
    print()

    rubric = func_localize_rubrics

    # Counters (thread-safe via lock)
    lock = threading.Lock()
    dim_scores: dict[str, list[int]] = {d: [] for d in DIMS}
    combined_avgs: list[float] = []
    ok_count = 0
    error_count = 0
    skip_count = 0
    done_count = len(already_done)

    # Load existing scores for tracking
    if already_done and output_file.exists():
        with open(output_file) as f:
            for line in f:
                rec = json.loads(line)
                if "scores" in rec:
                    ok_count += 1
                    for d in DIMS:
                        if d in rec["scores"]:
                            dim_scores[d].append(rec["scores"][d])
                    if "average" in rec:
                        combined_avgs.append(rec["average"])
                elif "error" in rec:
                    error_count += 1
                else:
                    skip_count += 1

    # Filter to pending records
    pending = [r for r in records if r["instance_id"] not in already_done]

    start_time = time.time()
    update_run_md(total, done_count, ok_count, error_count, skip_count, dim_scores, combined_avgs, model, start_time, workers)

    out_f = open(output_file, "a")
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(score_one, rec, rubric, model, api_key): rec["instance_id"]
                for rec in pending
            }
            for future in as_completed(futures):
                iid = futures[future]
                result = future.result()
                elapsed = result.pop("_elapsed", 0)

                with lock:
                    done_count += 1
                    if "scores" in result:
                        ok_count += 1
                        for d in DIMS:
                            if d in result["scores"]:
                                dim_scores[d].append(result["scores"][d])
                        combined_avgs.append(result["average"])
                        print(f"[{done_count}/{total}] {iid} OK scores={result['scores']} avg={result['average']:.2f} ({elapsed:.1f}s)")
                    elif "skip" in result:
                        skip_count += 1
                        print(f"[{done_count}/{total}] {iid} SKIP ({elapsed:.1f}s)")
                    else:
                        error_count += 1
                        err_msg = result.get("error", "unknown")
                        print(f"[{done_count}/{total}] {iid} ERROR: {err_msg} ({elapsed:.1f}s)")

                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()

                    if done_count % 10 == 0:
                        update_run_md(total, done_count, ok_count, error_count, skip_count, dim_scores, combined_avgs, model, start_time, workers)

    finally:
        out_f.close()

    update_run_md(total, done_count, ok_count, error_count, skip_count, dim_scores, combined_avgs, model, start_time, workers, finished=True)
    print()
    print("=" * 60)
    print("DONE")
    print(f"Success: {ok_count}, Errors: {error_count}, Skipped: {skip_count}")
    for d in DIMS:
        vals = dim_scores[d]
        if vals:
            print(f"  {d}: avg={sum(vals)/len(vals):.3f} (n={len(vals)})")
    if combined_avgs:
        print(f"  average_of_5: {sum(combined_avgs)/len(combined_avgs):.3f}")


if __name__ == "__main__":
    main()
