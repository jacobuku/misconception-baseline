"""One-off: strip Kaggle raw text (StudentExplanation, MC_Answer) from results/*.json.

The competition rules forbid redistributing train.csv, so per-row records keep
only ids, labels and predictions. Every other field and top-level block is
left untouched. Rewrites the files in place; safe to re-run.
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
FILES = ["llm_ambiguous.json", "llm_constrained.json"]
RAW_FIELDS = ("StudentExplanation", "MC_Answer")


def main():
    for name in FILES:
        path = RESULTS / name
        data = json.loads(path.read_text())
        removed = 0
        for pred in data["predictions"]:
            for field in RAW_FIELDS:
                if pred.pop(field, None) is not None:
                    removed += 1
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data['predictions'])} predictions, {removed} fields removed")


if __name__ == "__main__":
    main()
