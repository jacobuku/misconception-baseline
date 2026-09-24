"""Constrained-candidate variant of baseline_llm.py, on the identical 200 rows.

baseline_llm.py offered the model every label the (QuestionId, MC_Answer) cell
contained, so the true label was always somewhere in the candidate list. That is
an optimistic setup: it assumes the misconception taxonomy for this question and
option is already complete.

This variant simulates what a teacher actually has before the lesson - a short
list of misconceptions they anticipated. Per cell, only labels holding >= 10% of
the cell survive; everything rarer is not offered and is implicitly handled as
the cell's top-1 label ("not one of the anticipated misconceptions, treat as
default").

The cost is paid honestly: when a row's true label was pruned, no prediction can
match it and the row scores 0. Those rows are NOT excluded, NOT remapped, and
NOT given partial credit - they are the price of an incomplete taxonomy, and
reporting the score without them would be the whole point missed.

Everything else is shared with baseline_llm.py by import - the same sample, the
same fit/eval split, the same prompt layout, the same model and settings - so
the two runs differ only in which candidates are on the list.

Usage:
    python baseline_llm_constrained.py --dry-run
    python baseline_llm_constrained.py
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import time
from collections import Counter
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd

import baseline_llm as bl

ROOT = Path(__file__).resolve().parent
PREV_JSON = ROOT / "results" / "llm_ambiguous.json"
OUT_JSON = ROOT / "results" / "llm_constrained.json"

KEEP_SHARE = 0.10  # a label must hold this much of the cell to be anticipated

# bl.SYSTEM promises the correct label is always among the candidates. Under
# pruning that promise is false, so this one paragraph is rewritten. Leaving the
# original text would feed the model a claim this setup violates, which would
# confound the comparison rather than control it. Everything else is identical.
_OLD = """\
Every candidate label you are given is one that real students in this exact
question-and-option cell received, so the correct label IS in the list. Your job
is only to decide which candidate this particular explanation matches, judging
the reasoning the student actually wrote - not whether their final answer was
right, which is already fixed by the option they chose."""

_NEW = """\
The candidate labels are the outcomes the teacher anticipated for this exact
question-and-option cell. They are the only labels available, and they may not
cover this explanation - rarer outcomes were deliberately left off the list. If
none of the candidates truly fits, rank the closest ones anyway, judging the
reasoning the student actually wrote - not whether their final answer was right,
which is already fixed by the option they chose."""

assert _OLD in bl.SYSTEM, "bl.SYSTEM changed; re-check the constrained variant"
SYSTEM_CONSTRAINED = bl.SYSTEM.replace(_OLD, _NEW)


def keep_labels(cell: pd.DataFrame, share: float = KEEP_SHARE) -> list[str]:
    """Labels holding >= `share` of the cell, in prior order. Top-1 always survives."""
    counts = Counter(cell["label"])
    n = len(cell)
    kept = [lab for lab, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c / n >= share]
    return kept or [max(counts, key=lambda l: (counts[l], l))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args()

    # ---- rebuild the identical sample -------------------------------------- #
    df = bl.load()
    keys, cells = bl.ambiguous_cells(df, bl.AMBIGUITY_THRESHOLD)
    keyset = set(keys)
    in_amb = pd.Series(list(zip(df["QuestionId"], df["MC_Answer"])), index=df.index).isin(keyset)
    sub = df[in_amb]

    prev = json.loads(PREV_JSON.read_text())
    n = prev["config"]["n_eval"]
    eval_df = sub.sample(n, random_state=bl.SEED)

    prev_ids = [r["row_id"] for r in prev["predictions"]]
    if list(eval_df["row_id"]) != prev_ids:
        raise SystemExit("sample does not match baseline_llm.py's rows - not comparable")
    print(f"sample check: {n} row_ids identical to {PREV_JSON.name}  OK")

    fit = df.drop(index=eval_df.index)
    fit_cells = {k: g for k, g in fit.groupby(["QuestionId", "MC_Answer"], sort=False)}
    prev_by_id = {r["row_id"]: r for r in prev["predictions"]}

    # ---- prune candidates --------------------------------------------------- #
    prompts, pruned_stats = {}, []
    for idx, row in eval_df.iterrows():
        cell = fit_cells[(row["QuestionId"], row["MC_Answer"])]
        kept = keep_labels(cell)
        all_labs = prev_by_id[int(row["row_id"])]["candidates"]
        prompts[idx] = bl.build_prompt(row, cell, keep=kept) + (kept,)
        pruned_stats.append(
            {
                "idx": idx,
                "n_all": len(all_labs),
                "n_kept": len(kept),
                "dropped": [l for l in all_labs if l not in kept],
                "truth_pruned": row["label"] not in kept,
            }
        )

    n_doomed = sum(s["truth_pruned"] for s in pruned_stats)
    print(f"\n=== candidate pruning (keep share >= {KEEP_SHARE:.0%}) ===")
    print(f"  candidates/row : {np.mean([s['n_all'] for s in pruned_stats]):.2f} -> "
          f"{np.mean([s['n_kept'] for s in pruned_stats]):.2f}")
    print(f"  rows with >=1 label dropped : {sum(1 for s in pruned_stats if s['dropped'])}/{n}")
    print(f"  rows whose TRUE label was dropped : {n_doomed}/{n} "
          f"({n_doomed/n*100:.1f}%) -> these can only score 0")
    print(f"  => MAP@3 ceiling for this run : {1 - n_doomed/n:.4f}")
    drop_counts = collections.Counter(l for s in pruned_stats for l in s["dropped"])
    print("\n  most frequently dropped labels:")
    for lab, c in drop_counts.most_common(8):
        doomed = sum(1 for s in pruned_stats if s["truth_pruned"] and lab in s["dropped"]
                     and eval_df.loc[s["idx"], "label"] == lab)
        print(f"    {lab:<48} dropped in {c:>3} rows, was the truth in {doomed:>2}")

    if args.dry_run:
        idx = eval_df.index[0]
        prompt, cands, kept = prompts[idx]
        print("\n=== dry run: first row, constrained prompt ===")
        print(prompt)
        print(f"\n--- truth: {eval_df.loc[idx,'label']}  "
              f"was {prev_by_id[int(eval_df.loc[idx,'row_id'])]['candidates']} -> now {cands} ---")
        return

    # ---- run ---------------------------------------------------------------- #
    if not os.environ.get("ANTHROPIC_API_KEY"):
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()

    client = anthropic.Anthropic(max_retries=4)
    usage = bl.Usage()
    print(f"\n=== {n} rows -> {bl.MODEL} (effort={bl.EFFORT}, concurrency={bl.MAX_WORKERS}) ===")
    t0 = time.time()
    results, done = {}, 0

    def work(idx):
        prompt, cands, _ = prompts[idx]
        ranking, err = bl.ask(client, prompt, cands, usage, system=SYSTEM_CONSTRAINED)
        return idx, ranking, err, cands

    with cf.ThreadPoolExecutor(max_workers=bl.MAX_WORKERS) as pool:
        for idx, ranking, err, cands in pool.map(work, list(eval_df.index)):
            results[idx] = (ranking, err, cands)
            done += 1
            if done % 40 == 0 or done == n:
                print(f"  {done}/{n}  {time.time()-t0:.0f}s  ${usage.cost():.2f}")
    elapsed = time.time() - t0

    # ---- score --------------------------------------------------------------- #
    per_row, n_err = [], 0
    for (idx, row), st in zip(eval_df.iterrows(), pruned_stats):
        ranking, err, cands = results[idx]
        truth = row["label"]
        c_pred = cands[:bl.K]                       # constrained lookup baseline
        llm_pred = bl.pad(ranking, c_pred) if ranking else c_pred
        n_err += bool(err)
        prev_r = prev_by_id[int(row["row_id"])]
        per_row.append(
            {
                "row_id": int(row["row_id"]),
                "QuestionId": int(row["QuestionId"]),
                "truth": truth,
                "candidates_full": prev_r["candidates"],
                "candidates_kept": cands,
                "dropped": st["dropped"],
                "truth_pruned": bool(st["truth_pruned"]),
                "baseline_C_constrained_pred": c_pred,
                "llm_pred": llm_pred,
                "llm_raw_ranking": ranking,
                "baseline_C_constrained_ap": round(bl.apk(truth, c_pred), 4),
                "llm_ap": round(bl.apk(truth, llm_pred), 4),
                "prev_llm_pred": prev_r["llm_pred"],
                "prev_llm_ap": prev_r["llm_ap"],
                "prev_baseline_C_ap": prev_r["baseline_C_ap"],
                "delta_vs_prev_llm": round(bl.apk(truth, llm_pred) - prev_r["llm_ap"], 4),
                "error": err,
            }
        )

    def m(key, rows=per_row):
        return float(np.mean([r[key] for r in rows]))

    llm_map, cc_map = m("llm_ap"), m("baseline_C_constrained_ap")
    prev_llm_map, prev_c_map = m("prev_llm_ap"), m("prev_baseline_C_ap")
    llm_top1 = float(np.mean([r["llm_pred"][0] == r["truth"] for r in per_row]))
    llm_hit3 = float(np.mean([r["truth"] in r["llm_pred"] for r in per_row]))
    prev_top1 = float(np.mean([r["prev_llm_pred"][0] == r["truth"] for r in per_row]))
    prev_hit3 = float(np.mean([r["truth"] in r["prev_llm_pred"] for r in per_row]))

    survivors = [r for r in per_row if not r["truth_pruned"]]

    # by truth label
    by_lab, groups = [], collections.defaultdict(list)
    for r in per_row:
        groups[r["truth"]].append(r)
    for lab, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        by_lab.append(
            {
                "label": lab,
                "n": len(rs),
                "n_truth_pruned": sum(r["truth_pruned"] for r in rs),
                "prev_baseline_C": round(m("prev_baseline_C_ap", rs), 3),
                "prev_llm": round(m("prev_llm_ap", rs), 3),
                "constrained_llm": round(m("llm_ap", rs), 3),
                "delta": round(m("llm_ap", rs) - m("prev_llm_ap", rs), 3),
            }
        )

    tc = groups.get("True_Correct:NA", [])
    tc_report = {
        "n": len(tc),
        "n_truth_pruned": sum(r["truth_pruned"] for r in tc),
        "baseline_C_full": round(m("prev_baseline_C_ap", tc), 4),
        "llm_full_candidates": round(m("prev_llm_ap", tc), 4),
        "llm_constrained": round(m("llm_ap", tc), 4),
        "delta_constrained_vs_full": round(m("llm_ap", tc) - m("prev_llm_ap", tc), 4),
        "top1_full": round(float(np.mean([r["prev_llm_pred"][0] == r["truth"] for r in tc])), 4),
        "top1_constrained": round(float(np.mean([r["llm_pred"][0] == r["truth"] for r in tc])), 4),
    }

    out = {
        "config": {
            "model": bl.MODEL, "effort": bl.EFFORT, "thinking": "adaptive",
            "n_eval": n, "seed": bl.SEED, "keep_share": KEEP_SHARE,
            "ambiguity_threshold": bl.AMBIGUITY_THRESHOLD,
            "metric": f"MAP@{bl.K}",
            "same_rows_as": PREV_JSON.name,
            "system_prompt_deviation": (
                "bl.SYSTEM's 'the correct label IS in the list' guarantee is false "
                "under pruning and was rewritten; all other prompt text identical."
            ),
        },
        "pruning": {
            "mean_candidates_full": round(float(np.mean([s["n_all"] for s in pruned_stats])), 3),
            "mean_candidates_kept": round(float(np.mean([s["n_kept"] for s in pruned_stats])), 3),
            "rows_with_a_dropped_label": sum(1 for s in pruned_stats if s["dropped"]),
            "rows_with_truth_pruned": n_doomed,
            "map3_ceiling": round(1 - n_doomed / n, 4),
            "dropped_label_counts": dict(drop_counts.most_common()),
        },
        "scores": {
            "constrained_llm": {"map@3": round(llm_map, 4), "top1_acc": round(llm_top1, 4),
                                "top3_hit_rate": round(llm_hit3, 4)},
            "constrained_baseline_C": {"map@3": round(cc_map, 4)},
            "reference_full_candidates": {
                "llm": {"map@3": round(prev_llm_map, 4), "top1_acc": round(prev_top1, 4),
                        "top3_hit_rate": round(prev_hit3, 4)},
                "baseline_C": {"map@3": round(prev_c_map, 4)},
            },
            "delta_llm_constrained_vs_full": round(llm_map - prev_llm_map, 4),
            "on_survivable_rows_only": {
                "n": len(survivors),
                "constrained_llm_map@3": round(m("llm_ap", survivors), 4),
                "full_llm_map@3": round(m("prev_llm_ap", survivors), 4),
                "delta": round(m("llm_ap", survivors) - m("prev_llm_ap", survivors), 4),
            },
        },
        "by_truth_label": by_lab,
        "true_correct_na_report": tc_report,
        "cost": {
            "elapsed_sec": round(elapsed, 1), "api_calls": usage.calls, "errors": n_err,
            "input_tokens": usage.input, "output_tokens": usage.output,
            "usd_total": round(usage.cost(), 4),
        },
        "predictions": per_row,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    # ---- print --------------------------------------------------------------- #
    print(f"\n=== scores on the same {n} rows ===")
    print(f"  {'':34}{'MAP@3':>9}{'top-1':>9}{'top-3':>9}")
    print(f"  {'LLM, full candidates (prev)':34}{prev_llm_map:>9.4f}{prev_top1:>9.4f}{prev_hit3:>9.4f}")
    print(f"  {'LLM, constrained candidates':34}{llm_map:>9.4f}{llm_top1:>9.4f}{llm_hit3:>9.4f}")
    print(f"  {'delta':34}{llm_map-prev_llm_map:>+9.4f}{llm_top1-prev_top1:>+9.4f}{llm_hit3-prev_hit3:>+9.4f}")
    print(f"  {'baseline C, constrained':34}{cc_map:>9.4f}")
    print(f"  {'baseline C, full (prev)':34}{prev_c_map:>9.4f}")
    print(f"  ceiling imposed by pruning: {1-n_doomed/n:.4f}  "
          f"({n_doomed} rows can only score 0)")
    print(f"  on the {len(survivors)} survivable rows: constrained "
          f"{m('llm_ap', survivors):.4f} vs full {m('prev_llm_ap', survivors):.4f} "
          f"({m('llm_ap', survivors)-m('prev_llm_ap', survivors):+.4f})")

    print(f"\n=== by truth label (n >= 8) ===")
    print(f"  {'label':<42}{'n':>4}{'pruned':>7}{'C':>8}{'LLM':>8}{'cLLM':>8}{'delta':>8}")
    tail = []
    for b in by_lab:
        if b["n"] < 8:
            tail.append(b); continue
        print(f"  {b['label']:<42}{b['n']:>4}{b['n_truth_pruned']:>7}"
              f"{b['prev_baseline_C']:>8.3f}{b['prev_llm']:>8.3f}{b['constrained_llm']:>8.3f}{b['delta']:>+8.3f}")
    if tail:
        rs = [r for b in tail for r in groups[b["label"]]]
        print(f"  {'(all rarer labels)':<42}{len(rs):>4}{sum(r['truth_pruned'] for r in rs):>7}"
              f"{m('prev_baseline_C_ap', rs):>8.3f}{m('prev_llm_ap', rs):>8.3f}"
              f"{m('llm_ap', rs):>8.3f}{m('llm_ap', rs)-m('prev_llm_ap', rs):>+8.3f}")

    print(f"\n=== True_Correct:NA ===")
    print(f"  n = {tc_report['n']}   truth pruned in {tc_report['n_truth_pruned']} of them")
    print(f"  baseline C (full)        MAP@3 {tc_report['baseline_C_full']:.4f}")
    print(f"  LLM, full candidates     MAP@3 {tc_report['llm_full_candidates']:.4f}"
          f"   top-1 {tc_report['top1_full']:.4f}")
    print(f"  LLM, constrained         MAP@3 {tc_report['llm_constrained']:.4f}"
          f"   top-1 {tc_report['top1_constrained']:.4f}")
    print(f"  delta (constrained - full)     {tc_report['delta_constrained_vs_full']:+.4f}")

    print(f"\n  elapsed {elapsed:.1f}s   {usage.input:,} in / {usage.output:,} out   "
          f"${usage.cost():.4f}" + (f"   errors {n_err}" if n_err else ""))
    print(f"  wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
