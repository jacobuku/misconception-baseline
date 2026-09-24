"""Trivial baselines for MAP - Charting Student Math Misunderstandings.

Three baselines that deliberately IGNORE StudentExplanation. They answer the
question "how far does the label prior alone get you?", which is the floor any
real model has to clear.

    A) global      - every row gets the 3 globally most common combos
    B) question    - every row gets the 3 most common combos for its QuestionId
    C) qid+answer  - every row gets the 3 most common combos for its
                     (QuestionId, MC_Answer), padded from the question's top
                     combos when that cell has fewer than 3

Metric: MAP@3. Each row has exactly one true label, so AP@3 is 1/rank when the
label lands in the top 3 and 0 otherwise.

No API calls. Priors are fit on the training folds only.
"""

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "data" / "train.csv"
OUT_JSON = ROOT / "results" / "trivial.json"

N_SPLITS = 5
SEED = 42
K = 3

# Columns these baselines are allowed to touch. StudentExplanation is never
# read at all, so a stray reference would fail loudly rather than leak.
USED_COLS = ["row_id", "QuestionId", "MC_Answer", "Category", "Misconception"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV, usecols=USED_COLS)
    df["label"] = df["Category"].astype(str) + ":" + df["Misconception"].fillna("NA").astype(str)
    return df


def top_k(counter: Counter, k: int = K) -> list[str]:
    """Most common k labels, ties broken by label name for determinism."""
    return [lab for lab, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def pad(preds: list[str], *fallbacks: list[str], k: int = K) -> list[str]:
    """Fill preds up to k from the fallback lists, skipping duplicates."""
    out = list(preds)
    for fb in fallbacks:
        for lab in fb:
            if len(out) >= k:
                return out[:k]
            if lab not in out:
                out.append(lab)
    return out[:k]


def apk(actual: str, predicted: list[str], k: int = K) -> float:
    """Average precision at k for a single-relevant-item problem."""
    for i, p in enumerate(predicted[:k]):
        if p == actual:
            return 1.0 / (i + 1)
    return 0.0


def mapk(actuals: list[str], predictions: list[list[str]], k: int = K) -> float:
    return float(np.mean([apk(a, p, k) for a, p in zip(actuals, predictions)]))


def fit_priors(tr: pd.DataFrame) -> dict:
    """Fit all three priors on one training fold."""
    global_top = top_k(Counter(tr["label"]))

    by_q = {
        qid: top_k(Counter(g["label"]))
        for qid, g in tr.groupby("QuestionId", sort=False)
    }
    by_qa = {
        key: top_k(Counter(g["label"]))
        for key, g in tr.groupby(["QuestionId", "MC_Answer"], sort=False)
    }
    return {"global": global_top, "by_q": by_q, "by_qa": by_qa}


def predict(va: pd.DataFrame, priors: dict) -> dict[str, list[list[str]]]:
    """Produce top-3 predictions for each baseline on one validation fold."""
    g_top, by_q, by_qa = priors["global"], priors["by_q"], priors["by_qa"]

    preds = {"A_global": [], "B_question": [], "C_question_answer": []}
    for qid, ans in zip(va["QuestionId"].to_numpy(), va["MC_Answer"].to_numpy()):
        q_top = by_q.get(qid, [])
        qa_top = by_qa.get((qid, ans), [])

        preds["A_global"].append(g_top)
        # Unseen question -> global prior. Cannot happen with 15 questions,
        # but the padding keeps every row at exactly 3 predictions.
        preds["B_question"].append(pad(q_top, g_top))
        preds["C_question_answer"].append(pad(qa_top, q_top, g_top))
    return preds


def main() -> None:
    df = load_data()
    y = df["label"].to_numpy()

    # 14 of the 65 combos have <5 rows, so sklearn warns it cannot place one in
    # every fold. That is expected and does not invalidate the split.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds = list(skf.split(df, y))

    names = ["A_global", "B_question", "C_question_answer"]
    fold_scores: dict[str, list[float]] = {n: [] for n in names}
    # Coverage: how often the true label is in the top 3 at all, ignoring rank.
    fold_hits: dict[str, list[float]] = {n: [] for n in names}
    # Rank-1 accuracy, the half of MAP@3 that ranking quality actually moves.
    fold_top1: dict[str, list[float]] = {n: [] for n in names}

    for fold, (tr_idx, va_idx) in enumerate(folds):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        priors = fit_priors(tr)
        preds = predict(va, priors)
        actual = va["label"].tolist()

        for n in names:
            fold_scores[n].append(mapk(actual, preds[n]))
            fold_hits[n].append(
                float(np.mean([a in p[:K] for a, p in zip(actual, preds[n])]))
            )
            fold_top1[n].append(
                float(np.mean([p[0] == a for a, p in zip(actual, preds[n])]))
            )
        print(
            f"fold {fold}  "
            + "  ".join(f"{n}={fold_scores[n][-1]:.4f}" for n in names)
        )

    results = {
        "config": {
            "data": str(TRAIN_CSV.relative_to(ROOT)),
            "n_rows": int(len(df)),
            "n_classes": int(df["label"].nunique()),
            "n_splits": N_SPLITS,
            "seed": SEED,
            "stratify_on": "Category:Misconception",
            "metric": f"MAP@{K}",
            "features_used": ["QuestionId", "MC_Answer"],
            "features_forbidden": ["StudentExplanation"],
            "api_calls": 0,
        },
        "baselines": {
            n: {
                "description": desc,
                "fold_scores": [round(s, 6) for s in fold_scores[n]],
                "mean": round(float(np.mean(fold_scores[n])), 6),
                "std": round(float(np.std(fold_scores[n])), 6),
                "top1_acc_mean": round(float(np.mean(fold_top1[n])), 6),
                "top3_hit_rate_mean": round(float(np.mean(fold_hits[n])), 6),
            }
            for n, desc in zip(
                names,
                [
                    "3 globally most common Category:Misconception combos",
                    "3 most common combos within the row's QuestionId",
                    "3 most common combos within (QuestionId, MC_Answer), padded from the question's top combos",
                ],
            )
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")

    print()
    print(f"{'baseline':<20} {'MAP@3':>8} {'std':>8} {'top-1':>8} {'top3 hit':>9}")
    print("-" * 57)
    for n in names:
        b = results["baselines"][n]
        print(f"{n:<20} {b['mean']:>8.4f} {b['std']:>8.4f} "
              f"{b['top1_acc_mean']:>8.4f} {b['top3_hit_rate_mean']:>9.4f}")
    print()
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
