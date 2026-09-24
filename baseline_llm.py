"""LLM baseline on the ambiguous subset of MAP - Charting Student Math Misunderstandings.

baseline_trivial.py showed that a (QuestionId, MC_Answer) lookup table already
scores MAP@3 = 0.839 without reading a single word the student wrote. Running an
LLM over the full training set would mostly pay to re-derive that lookup table.

So this script only runs where the lookup table cannot decide: cells whose most
common label holds less than AMBIGUITY_THRESHOLD of the cell. Inside those cells
the StudentExplanation is the only signal left, which makes them the honest test
of whether an LLM adds anything.

Pipeline:
  1. find the ambiguous cells and report their coverage
  2. sample N rows from them (seed=42), ask Claude to rank that cell's candidate
     labels, structured-output-constrained to those exact labels
  3. score LLM vs the C lookup baseline on the same rows, MAP@3

Leakage: the sampled rows are removed from the data used to build the lookup
table, the candidate lists and the few-shot examples, so neither side has seen
the rows it is scored on.

Usage:
    python baseline_llm.py --dry-run     # step 1 + prompt preview, no API calls
    python baseline_llm.py -n 20         # smoke test
    python baseline_llm.py -n 200        # full run
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "data" / "train.csv"
OUT_JSON = ROOT / "results" / "llm_ambiguous.json"

MODEL = "claude-sonnet-4-6"
# $ per 1M tokens for claude-sonnet-4-6.
PRICE_IN, PRICE_OUT = 3.00, 15.00
PRICE_CACHE_READ, PRICE_CACHE_WRITE = 0.30, 3.75

AMBIGUITY_THRESHOLD = 0.80
N_EXAMPLES_PER_LABEL = 2
SEED = 42
K = 3
MAX_WORKERS = 10
MAX_TOKENS = 4000
EFFORT = "medium"

SYSTEM = """\
You are labelling how UK secondary-school students explain their answer to a \
multiple-choice maths question.

Each label has the form `Category:Misconception`.

The Category has two halves, `<AnswerCorrect>_<ExplanationVerdict>`:
  True_*  - the student picked the correct option
  False_* - the student picked an incorrect option
  *_Correct       - the explanation shows correct, complete reasoning
  *_Misconception - the explanation reveals a specific, recognised misconception
  *_Neither       - the explanation is neither correct reasoning nor a known
                    misconception: it may be vague, circular, off-topic, a guess,
                    a restatement of the answer, or simply too thin to judge

`:NA` means no misconception was recorded.

Every candidate label you are given is one that real students in this exact
question-and-option cell received, so the correct label IS in the list. Your job
is only to decide which candidate this particular explanation matches, judging
the reasoning the student actually wrote - not whether their final answer was
right, which is already fixed by the option they chose.

Read the worked examples for each candidate carefully: they define what that
label means for this specific question. Rank the candidates from most to least
likely. Return only the ranking."""


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV)
    df["label"] = df["Category"].astype(str) + ":" + df["Misconception"].fillna("NA").astype(str)
    return df


def ambiguous_cells(df: pd.DataFrame, threshold: float) -> tuple[list[tuple], pd.DataFrame]:
    """Cells whose most common label holds less than `threshold` of the cell."""
    rows = []
    for (qid, ans), g in df.groupby(["QuestionId", "MC_Answer"], sort=False):
        counts = g["label"].value_counts()
        rows.append(
            {
                "QuestionId": qid,
                "MC_Answer": ans,
                "n_rows": len(g),
                "n_labels": len(counts),
                "top_label": counts.index[0],
                "top_share": counts.iloc[0] / len(g),
            }
        )
    cells = pd.DataFrame(rows).sort_values("top_share")
    amb = cells[cells["top_share"] < threshold]
    keys = list(zip(amb["QuestionId"], amb["MC_Answer"]))
    return keys, cells


def top_k(counter: Counter, k: int = K) -> list[str]:
    return [lab for lab, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def apk(actual: str, predicted: list[str], k: int = K) -> float:
    for i, p in enumerate(predicted[:k]):
        if p == actual:
            return 1.0 / (i + 1)
    return 0.0


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #
def strip_latex(s: str) -> str:
    """Make \\( \\frac{1}{3} \\) readable without losing the maths."""
    s = re.sub(r"\\[()\[\]]", "", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1/\2)", s)
    s = re.sub(r"\\(times|div|square|pi)", lambda m: {"times": "x", "div": "/", "square": "[ ]", "pi": "pi"}[m.group(1)], s)
    return re.sub(r"\s+", " ", s).strip()


def build_prompt(
    row: pd.Series, cell: pd.DataFrame, keep: list[str] | None = None
) -> tuple[str, list[str]]:
    """Render one row's prompt. Returns (prompt, candidate labels in prior order).

    `keep` restricts which of the cell's labels are offered as candidates;
    everything else about the rendered prompt is unchanged. Default None keeps
    every label the cell contains.
    """
    counts = Counter(cell["label"])
    # Prior order, so the model sees the lookup-table ranking it has to beat.
    candidates = [lab for lab, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    if keep is not None:
        candidates = [lab for lab in candidates if lab in set(keep)]

    parts = [
        "## Question",
        strip_latex(str(row["QuestionText"])),
        "",
        "## The option this student chose",
        strip_latex(str(row["MC_Answer"])),
        "",
        f"## Candidate labels ({len(candidates)}), with real examples from students "
        "who chose this same option",
        "",
    ]
    for lab in candidates:
        ex = cell[cell["label"] == lab]
        ex = ex.sample(min(N_EXAMPLES_PER_LABEL, len(ex)), random_state=SEED)
        parts.append(
            f"### {lab}   ({counts[lab]} of {len(cell)} students who chose this option)"
        )
        for e in ex["StudentExplanation"]:
            parts.append(f'  - "{e}"')
        parts.append("")

    parts += [
        "## The explanation to label",
        f'"{row["StudentExplanation"]}"',
        "",
        f"Rank the {min(K, len(candidates))} most likely candidate labels for this "
        "explanation, most likely first. Use the candidate strings exactly.",
    ]
    return "\n".join(parts), candidates


# --------------------------------------------------------------------------- #
# api
# --------------------------------------------------------------------------- #
class Usage:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.input = self.output = self.cache_read = self.cache_write = 0
        self.calls = self.errors = 0

    def add(self, u) -> None:
        with self.lock:
            self.calls += 1
            self.input += u.input_tokens
            self.output += u.output_tokens
            self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
            self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0

    def cost(self) -> float:
        return (
            self.input * PRICE_IN
            + self.output * PRICE_OUT
            + self.cache_read * PRICE_CACHE_READ
            + self.cache_write * PRICE_CACHE_WRITE
        ) / 1e6


def ask(
    client: anthropic.Anthropic,
    prompt: str,
    candidates: list[str],
    usage: Usage,
    system: str = SYSTEM,
) -> tuple[list[str], str | None]:
    """One call. Returns (ranking, error). Ranking is enum-constrained to candidates."""
    schema = {
        "type": "object",
        "properties": {
            "ranking": {
                "type": "array",
                "items": {"type": "string", "enum": candidates},
                # The API supports neither minItems > 1 nor maxItems, so the
                # schema enforces only membership (the part that matters); the
                # length is handled by the instruction plus dedupe/truncate/pad.
            }
        },
        "required": ["ranking"],
        "additionalProperties": False,
    }
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
        )
    except anthropic.BadRequestError as e:
        return [], f"BadRequest: {e.message}"
    except anthropic.RateLimitError:
        return [], "RateLimit (exhausted SDK retries)"
    except anthropic.APIStatusError as e:
        return [], f"APIStatus {e.status_code}: {e.message}"
    except anthropic.APIConnectionError as e:
        return [], f"Connection: {e}"

    usage.add(resp.usage)
    if resp.stop_reason == "refusal":
        return [], f"refusal: {getattr(resp.stop_details, 'category', None)}"
    if resp.stop_reason == "max_tokens":
        return [], "max_tokens"

    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        return [], "no text block"
    try:
        ranking = json.loads(text)["ranking"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return [], f"parse: {e}"

    # The enum constrains membership only - dedupe and truncate here.
    seen, out = set(), []
    for lab in ranking:
        if lab not in seen:
            seen.add(lab)
            out.append(lab)
    return out[:K], None


def pad(preds: list[str], fallback: list[str], k: int = K) -> list[str]:
    out = list(preds)
    for lab in fallback:
        if len(out) >= k:
            break
        if lab not in out:
            out.append(lab)
    return out[:k]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200, help="rows to evaluate")
    ap.add_argument("--dry-run", action="store_true", help="step 1 + one prompt, no API calls")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args()

    df = load()

    # ---- step 1: define the ambiguous subset ---------------------------------
    keys, cells = ambiguous_cells(df, AMBIGUITY_THRESHOLD)
    keyset = set(keys)
    in_amb = pd.Series(list(zip(df["QuestionId"], df["MC_Answer"])), index=df.index).isin(keyset)
    sub = df[in_amb]

    print(f"=== step 1: ambiguous cells (top label < {AMBIGUITY_THRESHOLD:.0%}) ===")
    print(f"  cells total          : {len(cells)}")
    print(f"  cells ambiguous      : {len(keys)}  ({len(keys)/len(cells)*100:.1f}%)")
    print(f"  rows covered         : {len(sub):,}  ({len(sub)/len(df)*100:.1f}% of train)")
    print(f"  labels per amb. cell : min {cells.loc[cells.index.isin(cells[cells.top_share < AMBIGUITY_THRESHOLD].index), 'n_labels'].min()}"
          f"  max {cells[cells.top_share < AMBIGUITY_THRESHOLD]['n_labels'].max()}")
    print()
    print("  the 10 least decidable cells:")
    print("  " + f"{'QuestionId':>10} {'top share':>10} {'labels':>7} {'rows':>6}  MC_Answer")
    for _, c in cells.head(10).iterrows():
        print(f"  {c['QuestionId']:>10} {c['top_share']:>9.1%} {c['n_labels']:>7} {c['n_rows']:>6}  {strip_latex(str(c['MC_Answer']))[:34]}")
    print()

    if len(sub) < args.n:
        sys.exit(f"subset has only {len(sub)} rows, cannot sample {args.n}")

    # ---- step 2: sample, then fit everything else on the remainder -----------
    eval_df = sub.sample(args.n, random_state=SEED)
    fit = df.drop(index=eval_df.index)
    fit_cells = {k: g for k, g in fit.groupby(["QuestionId", "MC_Answer"], sort=False)}

    prompts = {}
    for idx, row in eval_df.iterrows():
        cell = fit_cells[(row["QuestionId"], row["MC_Answer"])]
        prompts[idx] = build_prompt(row, cell)

    if args.dry_run:
        idx = eval_df.index[0]
        prompt, cands = prompts[idx]
        print("=== dry run: prompt for the first sampled row ===")
        print(f"[system: {len(SYSTEM)} chars]\n")
        print(prompt)
        print(f"\n--- truth: {eval_df.loc[idx, 'label']}   candidates: {len(cands)} ---")
        est_in = np.mean([len(p) for p, _ in prompts.values()]) / 3.5 + len(SYSTEM) / 3.5
        print(f"\nrough estimate for {args.n} rows: ~{est_in:.0f} input tok/row, "
              f"~${args.n * (est_in * PRICE_IN + 700 * PRICE_OUT) / 1e6:.2f} "
              f"(assuming ~700 output tok/row with thinking)")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()

    client = anthropic.Anthropic(max_retries=4)
    usage = Usage()

    print(f"=== step 2: {args.n} rows -> {MODEL} (effort={EFFORT}, concurrency={MAX_WORKERS}) ===")
    t0 = time.time()
    results: dict = {}
    done = 0

    def work(idx):
        prompt, cands = prompts[idx]
        ranking, err = ask(client, prompt, cands, usage)
        return idx, ranking, err, cands

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for idx, ranking, err, cands in pool.map(work, list(eval_df.index)):
            results[idx] = (ranking, err, cands)
            done += 1
            if done % 20 == 0 or done == args.n:
                print(f"  {done}/{args.n}  {time.time()-t0:.0f}s  ${usage.cost():.2f}")
    elapsed = time.time() - t0

    # ---- step 3: score -------------------------------------------------------
    per_row, c_scores, llm_scores, n_err = [], [], [], 0
    for idx, row in eval_df.iterrows():
        ranking, err, cands = results[idx]
        truth = row["label"]
        c_pred = cands[:K]                       # C baseline: the cell's top-3
        llm_pred = pad(ranking, c_pred) if ranking else c_pred
        if err:
            n_err += 1
        c_scores.append(apk(truth, c_pred))
        llm_scores.append(apk(truth, llm_pred))
        per_row.append(
            {
                "row_id": int(row["row_id"]),
                "QuestionId": int(row["QuestionId"]),
                "truth": truth,
                "candidates": cands,
                "baseline_C_pred": c_pred,
                "llm_pred": llm_pred,
                "llm_raw_ranking": ranking,
                "baseline_C_ap": round(c_scores[-1], 4),
                "llm_ap": round(llm_scores[-1], 4),
                "delta": round(llm_scores[-1] - c_scores[-1], 4),
                "error": err,
            }
        )

    c_map, llm_map = float(np.mean(c_scores)), float(np.mean(llm_scores))
    c_top1 = float(np.mean([r["baseline_C_pred"][0] == r["truth"] for r in per_row]))
    llm_top1 = float(np.mean([r["llm_pred"][0] == r["truth"] for r in per_row]))
    hit3 = float(np.mean([r["truth"] in r["llm_pred"] for r in per_row]))

    out = {
        "config": {
            "model": MODEL,
            "effort": EFFORT,
            "thinking": "adaptive",
            "n_eval": args.n,
            "seed": SEED,
            "ambiguity_threshold": AMBIGUITY_THRESHOLD,
            "examples_per_label": N_EXAMPLES_PER_LABEL,
            "concurrency": MAX_WORKERS,
            "metric": f"MAP@{K}",
        },
        "subset": {
            "cells_total": int(len(cells)),
            "cells_ambiguous": int(len(keys)),
            "rows_in_ambiguous_cells": int(len(sub)),
            "pct_of_train": round(len(sub) / len(df) * 100, 2),
        },
        "scores": {
            "baseline_C": {"map@3": round(c_map, 4), "top1_acc": round(c_top1, 4)},
            "llm": {
                "map@3": round(llm_map, 4),
                "top1_acc": round(llm_top1, 4),
                "top3_hit_rate": round(hit3, 4),
            },
            "delta_map@3": round(llm_map - c_map, 4),
            "rows_llm_better": sum(1 for r in per_row if r["delta"] > 0),
            "rows_llm_worse": sum(1 for r in per_row if r["delta"] < 0),
            "rows_tied": sum(1 for r in per_row if r["delta"] == 0),
        },
        "cost": {
            "elapsed_sec": round(elapsed, 1),
            "api_calls": usage.calls,
            "errors": n_err,
            "input_tokens": usage.input,
            "output_tokens": usage.output,
            "usd_total": round(usage.cost(), 4),
            "usd_per_row": round(usage.cost() / max(args.n, 1), 5),
        },
        "predictions": per_row,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    print()
    print(f"=== step 3: {args.n} rows from the ambiguous subset ===")
    print(f"  {'':16} {'MAP@3':>8} {'top-1':>8}")
    print(f"  {'baseline C':16} {c_map:>8.4f} {c_top1:>8.4f}")
    print(f"  {'LLM':16} {llm_map:>8.4f} {llm_top1:>8.4f}")
    print(f"  {'delta':16} {llm_map-c_map:>+8.4f} {llm_top1-c_top1:>+8.4f}")
    print(f"  better/worse/tied: {out['scores']['rows_llm_better']}/"
          f"{out['scores']['rows_llm_worse']}/{out['scores']['rows_tied']}")
    print(f"  LLM top-3 hit rate: {hit3:.4f}")
    print()
    print(f"  elapsed  : {elapsed:.1f}s  ({elapsed/max(args.n,1):.2f}s/row at concurrency {MAX_WORKERS})")
    print(f"  tokens   : {usage.input:,} in / {usage.output:,} out over {usage.calls} calls")
    print(f"  cost     : ${usage.cost():.4f}  (${usage.cost()/max(args.n,1):.5f}/row)")
    if n_err:
        print(f"  errors   : {n_err} (fell back to baseline C)")
    print(f"  wrote {args.out.relative_to(ROOT)}")
    full = usage.cost() / max(args.n, 1) * len(sub)
    print(f"  extrapolated to all {len(sub):,} ambiguous rows: ~${full:.2f}")


if __name__ == "__main__":
    main()
