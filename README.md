# Misconception Baseline

A boundary-probing study of the Kaggle competition
**[MAP — Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)**.
The goal was never to climb the leaderboard. It was to answer one question:
**how much information does a student's written explanation actually carry in
this task, once you account for everything you could have known without reading
it?**

The short answer: less than the framing of the competition suggests. A lookup
table that never reads a single word scores MAP@3 = 0.8393.

---

## Data

- **Source:** Kaggle, *MAP — Charting Student Math Misunderstandings*. Hosted by
  Vanderbilt University, The Learning Agency and Kaggle; data provided by
  **Eedi**; supported by the Gates Foundation and the Walton Family Foundation.
  The responses are from students in **grades 4–8**.
- **Size:** 36,696 rows, 7 columns, **15 distinct questions**, 4 options each.
- **Label:** `Category:Misconception`, a single string per row — **65 distinct
  combinations** observed in `train.csv`. `Category` is
  `<AnswerCorrect>_<ExplanationVerdict>` (e.g. `True_Correct`,
  `False_Misconception`); `Misconception` is `NA` for 73.1% of rows.
- **Metric:** MAP@3 — each row has exactly one true label, so a row scores
  `1/rank` if the label appears in the top 3 and 0 otherwise.

**The data is not redistributable.** Competition rules prohibit republishing it,
so `data/` is in `.gitignore` and nothing under it is committed. To reproduce,
download `train.csv` from the competition page into `data/`.

`EXPLORE.md` (committed) contains the full exploratory profile — label
distributions, the long tail, text-length statistics, sample rows, and the data
quality problems listed under *Honest limitations*.

---

## What I measured

### 1. Three zero-cost lookup baselines that deliberately ignore the text

Before asking what a model adds, you need to know what the label prior alone is
worth. Three baselines, 5-fold stratified CV (seed 42), `StudentExplanation`
excluded at load time so it cannot leak:

- **A — global:** every row gets the 3 globally most common labels.
- **B — per question:** the 3 most common labels for that `QuestionId`.
- **C — per (question, option):** the 3 most common labels for that
  `(QuestionId, MC_Answer)` cell, padded from the question's top labels.

C is the interesting one. With 15 questions × 4 options there are only 60 cells,
and the option a student picks already fixes the `True_`/`False_` half of the
label and strongly constrains which misconception is plausible.

### 2. What an LLM adds where the lookup table cannot decide

Running an LLM over the full training set would mostly pay to re-derive C. So
the LLM was evaluated only on **ambiguous cells** — those whose most common
label holds less than 80% of the cell. 50 of the 60 cells qualify, covering
31,115 rows (84.8% of train).

200 rows were sampled from that subset (seed 42) and given to
`claude-sonnet-4-6` with the question, the chosen option, every label that cell
contains as an explicit candidate (2 real student examples each), and the
explanation to classify. Structured output constrained the response to an
enum of that cell's candidates, so the model cannot invent a label. The 200
sampled rows are removed from the data used to build the lookup table, the
candidate lists, and the few-shot examples — neither side has seen the rows it
is scored on.

### 3. What narrowing the candidate set costs

The setup above is optimistic: it hands the model every label the cell
contains, so the true label is always somewhere on the list. That assumes the
misconception taxonomy is already complete.

A teacher before a lesson has something weaker — a short list of misconceptions
they anticipated. To simulate that, the same 200 rows were re-run with
candidates pruned to labels holding **≥ 10% of their cell**; everything rarer is
not offered and is implicitly handled as the cell's top-1 label. Candidates per
row fell from 4.26 to 2.105.

When a row's true label was pruned, no prediction can match it and the row
scores 0. Those rows are not excluded, not remapped, and not given partial
credit — reporting the score without them would miss the entire point. 8 of the
200 rows (4.0%) are in that position, which caps this run's MAP@3 at **0.9600**.

---

## Results

All figures below are read from `results/*.json`.

| Approach | Reads the explanation? | Eval set | MAP@3 | top-1 | top-3 hit |
|---|---|---|---|---|---|
| A — global top-3 | no | 36,696 (5-fold) | 0.5403 | 0.4034 | 0.7251 |
| B — per question | no | 36,696 (5-fold) | 0.5843 | 0.4202 | 0.7995 |
| **C — per (question, option)** | no | 36,696 (5-fold) | **0.8393** | 0.6965 | **0.9901** |
| C, on the ambiguous subset | no | 200 | 0.8142 | 0.6600 | 0.9750 |
| **LLM, full candidates** | yes | 200 | **0.8758** | 0.7800 | 0.9850 |
| LLM, pruned candidates | yes | 200 | 0.8450 | 0.7350 | 0.9600 |
| C, pruned candidates | no | 200 | 0.8092 | 0.6600 | 0.9600 |

Fold-to-fold standard deviation for A/B/C is ≤ 0.0018. The LLM rows are a single
pass over 200 rows and carry correspondingly wider uncertainty.

Broken out by true label, on the same 200 rows (AP@3 averaged within each group):

| True label | n | truth pruned | C (full) | LLM (full) | LLM (pruned) |
|---|---|---|---|---|---|
| `True_Correct:NA` | 62 | 0 | 1.000 | 0.882 | 0.892 |
| `True_Neither:NA` | 33 | 0 | 0.500 | 0.843 | 0.848 |
| `False_Neither:NA` | 32 | 0 | 0.672 | 0.745 | 0.755 |
| `False_Misconception:Additive` | 10 | 0 | 1.000 | 1.000 | 1.000 |
| all rarer labels | 63 | 8 | 0.839 | 0.934 | 0.817 |

---

## Findings

### 1. The floor is 0.8393, and it never reads the text

A lookup on `(QuestionId, MC_Answer)` scores MAP@3 = 0.8393 with a **99.0% top-3
hit rate**. The correct label is almost always already in the top 3; what
remains is a ranking problem, not a retrieval problem.

Per the host's own case study, top-performing models reached MAP@3 **above
0.94** ([The Learning Agency, "Case Study: Math Misconceptions
Competition"](https://the-learning-agency.com/the-cutting-ed/article/case-study-math-misconceptions-competition/)
— 1,850+ teams, ~40,000 submissions). Against that, the text-blind lookup table
already recovers **~89% of the winning score**, and everything the written
explanation contributes is the remaining **≥ 0.10 MAP@3** — at least a quarter
of the distance from a constant predictor (0.5403) to the top.

Both of those are *lower* bounds, not caps: "above 0.94" pins a floor under the
ceiling, so a higher true winning score widens the text's share rather than
narrowing it. They also set a leaderboard test score against a train-set CV
score, which are not the same measurement. Treat them as an order of magnitude,
not a budget. The robust claim needs neither number: **the floor is 0.8393 at
99.0% top-3 recall, so any result in this task should be quoted against that,
not against zero.**

### 2. The LLM's gain is not uniform — it is a residual of two opposing forces

On the ambiguous subset the LLM beats the lookup table by **+0.0617**
(0.8758 vs 0.8142), winning 46 rows, losing 23, tying 131. But the aggregate
hides the structure:

- `True_Neither:NA`: **+0.343** (0.843 vs 0.500) — a large, real gain.
- `True_Correct:NA`: **−0.118** (0.882 vs 1.000) — the lookup table was perfect
  on these 62 rows and the LLM broke it.

The headline +0.06 is what survives after those cancel. Reporting only the
aggregate would hide the fact that reading the text makes one large class
strictly worse.

### 3. The model systematically over-attributes misconceptions

The failures on `True_Correct:NA` are one-directional: a student writes a
colloquial but valid justification, and the model reads a misconception into it.

```
"i think this because two negatives make a positive."
  truth      True_Correct:NA
  LLM (full) True_Misconception:Positive, True_Misconception:Tacking, True_Neither:NA   → AP 0.0

"all minuses means you add i think"
  truth      True_Correct:NA
  LLM (full) True_Neither:NA, True_Misconception:Positive, True_Misconception:Tacking   → AP 0.0
```

*Quoted as analysis examples under fair use; the dataset is not redistributed here.*

The natural fix is to take the tempting misconception labels off the list — but
that is not what happens. Across the 62 `True_Correct:NA` rows, pruning moved
the model's incorrect top-1 picks *from* scattered misconception labels (3 rows)
*to* `True_Neither:NA` (9 → 12 rows). Top-1 accuracy on the class went **down**,
0.8065 → 0.7903, even as MAP@3 edged up.

So the model is not being lured by rare misconception labels. It systematically
declines to call an informally-worded correct explanation "Correct". **That is a
`Correct` / `Neither` annotation-boundary problem, not a candidate-set problem**
— and it is the same boundary where the dataset's own annotators contradict each
other (see *Honest limitations*).

### 4. Narrowing the candidate set costs almost no discriminative power

Halving the candidate list (4.26 → 2.105 per row) drops MAP@3 from 0.8758 to
0.8450, a loss of 0.0308. Nearly all of it is structural, not cognitive:

- On the **192 rows whose true label survived pruning: −0.0017.** Effectively
  zero. All four major label groups actually improved slightly.
- The 8 rows whose true label was pruned scored 0.729 on average before; losing
  them costs 0.0292 of the 0.0308.

**A teacher's short, incomplete list of anticipated misconceptions is not what
limits this system.** What limits it is that 4% of students produce something
outside the list entirely — and no amount of model quality recovers those.

---

## Honest limitations

- **15 questions.** Train and test overlap on them, so per-question priors work
  well here and local CV says nothing about generalizing to unseen questions.
  This is the single biggest caveat: the 0.8393 floor is partly an artifact of
  a small, fixed question set.
- **61 annotation conflicts.** 580 groups of rows share an identical
  `(QuestionId, MC_Answer, StudentExplanation)`; **61 of them (10.5%) carry more
  than one label.** Identical input, different ground truth. That is a hard
  ceiling on any CV score, and the conflicts concentrate on exactly the
  `Correct` / `Neither` / `Misconception` boundary that Finding 3 is about.
- **`Wrong_Fraction` vs `Wrong_fraction`** differ only in case and are treated
  as two distinct classes. They are bound to different questions (31777 and
  33471), so merging them would change both the class count and the per-question
  candidate sets. **Left unmerged** — flagged, not fixed.
- **The LLM evaluation is 200 rows and $1.46 total.** It is a probe, not a
  benchmark. Single pass, no repeats, no confidence intervals, one prompt
  design, one model.
- **No fine-tuning, no ensembling, no test-set submission.** None of these
  numbers are directly comparable to leaderboard scores; they are CV and
  subset numbers measured for a different purpose.
- The pruned run rewrites one paragraph of the system prompt — the original
  guarantees the correct label is among the candidates, which pruning makes
  false. All other prompt text is identical. Recorded in
  `results/llm_constrained.json` under `config.system_prompt_deviation`.

---

## Reproduce

```bash
python -m venv .venv && ./.venv/bin/pip install pandas numpy scikit-learn anthropic
# download train.csv from the competition page into data/
```

The per-row records in `results/` have `StudentExplanation` and `MC_Answer`
stripped (competition rules forbid redistributing the data); they keep only
`row_id`, labels and predictions. To see the original text for any row, or to
re-run anything, download `train.csv` from Kaggle yourself and join on `row_id`.

| Step | Command | Time | Cost |
|---|---|---|---|
| Exploratory profile | *(written to `EXPLORE.md`)* | — | free |
| Lookup baselines | `python baseline_trivial.py` | ~5 s | free |
| LLM on ambiguous cells | `python baseline_llm.py -n 200` | 129 s | $0.86 |
| LLM, pruned candidates | `python baseline_llm_constrained.py` | 110 s | $0.60 |

The last two require `ANTHROPIC_API_KEY` (read from the environment, or from a
local `.env`). Both accept `--dry-run`, which prints the subset analysis and a
rendered prompt without making any API call — start there.

`baseline_llm_constrained.py` asserts that its 200 `row_id`s match
`results/llm_ambiguous.json` exactly and exits if they do not, so the two runs
are always comparable. Run `baseline_llm.py` first.

Outputs land in `results/` as `trivial.json`, `llm_ambiguous.json`, and
`llm_constrained.json`; the latter two include every per-row prediction,
candidate list, and ground truth.
