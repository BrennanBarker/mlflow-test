Plan:

- Generate a dataset by supplementing gold (source, summary) examples with corrupted ones, using an LLM instructed to corrupt the example summaries in specific ways.  In this way we produce labels of "true" / "false", corresponding to whether the data was faithful (not corrupted), along with metadata including a specific perterbation and the category of perterbation.  The former is useful as rationale for a judge evaluator, the latter for grouping runs to identify trends in judge behavior (missing specific categories more)

  - The corruption generator model should be a different (and ideally stronger) model family than whatever we land on as the judge, so the two don't share blind spots -- otherwise "hard" cases risk being hard only in ways the judge model already handles well.

  - Corruptions are not currently verified beyond "non-empty" and "not a no-op" (see `dataset/validate.py`). Since the judge evaluator trusts these labels as ground truth, add a verification step -- e.g. a second LLM check (or human spot-check of a sample) confirming the corruption actually broke faithfulness as instructed, and that it was a minimal single-fact change -- before the dataset is used for evaluation. Not yet implemented.

  - The current dev sample size (`--n-samples` default in `judge-eval generate-dataset`) is a scaffold only; once corruption type x target is factored in, a meaningfully larger sample will be needed to get trustworthy per-category accuracy numbers.

- Evaluate the judge by feeding the dataset through the judge to produce output, that output being in turn judged by a Judge Evaluator.  The Judge Evaluator's task is to identify whether the Judge succeeded or failed at identifying unfaithful examples.  The task is made simple by the labeled data produced by the data generation process; there are four possible states that the evaluator will need to consider when comparing the Judge's verdict to the label:
  1. Judge value: Yes, Label: No -- the judge has missed at least one example of corruption; the evaluator judges this a failure automatically and leverages the description of the induced corruption as rationale. (We assume that the data generation process succeeded at its task.  We accept the risk that the induced corruption has masked another corruption on the basis that even a single missed example is problematic) Verdict: Falure
  2. Judge value: No, Label: Yes -- the judge claims to have found an example of corruption that was not induced; this is possibly spurious and possibly a reflection of a natural corruption in the gold data -- example is preliminarily marked as a judge failure but flagged for human review
  3. Judge value: Yes, Label: Yes -- the judge sees no hallucinations in an uncorrupted example; we assume that this is accurate, verdict: Success
  4. Judge value: No, Label: No -- the judge identifies a corruption in a file which had an induced corruption.  It's possible that the judge is identifying some other feature of the example and is thus "right for the wrong reason" -- or perhaps even "right for the right reason, just not the reason we expected"; an LLM is invoked to compare the rationales for each -- if the rationales match, verdict is success, if the rationales differ, mark as failure and flag for human review.


- Before entering an optimization loop, hold out a fixed evaluation split that is never touched during prompt optimization. Otherwise optimizing the judge's prompt against the same dataset used to score it risks overfitting to artifacts of our own corruption generator rather than improving real hallucination detection.

- If the initial evaluation shows an insufficiently capable Judge or one that is prohibitive on cost or resources, enter an optimization process.  To increase scores, conduct a prompt optimization (using the same scoring mechanism, applied only to the non-held-out split), and/or move to a costly model and optimize. To decrease cost, move to a less costly model, evaluate, and optimize prompt as necessary.

## Layout

```
src/judge_eval/
  cli.py            entry point: `judge-eval generate-dataset` / `judge-eval run-evaluation`
  common/           shared oracle-call plumbing (model tiers, structured tool-call helper, mlflow tracking setup)
  dataset/          generation-side pipeline (corruption generation/verification, shared types)
  harness/          judge-agnostic evaluation harness (the four-case Scorer, mlflow.genai.evaluate orchestration)
  judges/           pluggable judges under test (today: the faithfulness judge)
data/
  original/xsum/    gold (document, summary) pairs (gitignored)
  generated/        generated dataset (gitignored)
```

Install once (editable, so code changes take effect immediately):

```
pip install -e .
```

Every `judge-eval` invocation is tracked as an MLflow run (local `mlflow.db` by default -- run `mlflow ui` to browse). Dataset generation runs land in the `judge-eval-dataset-generation` experiment; judge evaluation runs land in `judge-eval-judge-evaluation`, kept separate since the two run at very different frequencies and you'll usually only want to look at one at a time. `mlflow.openai.autolog()` and `mlflow.litellm.autolog()` are both enabled up front, so every LLM call made by either pipeline -- oracle calls (raw openai client) and the judge-under-test's own calls (routed through litellm by `make_judge`/`is_grounded`) alike -- shows up as a trace on its run.

## Usage

Two commands, run at very different frequencies:

**1. Generate the dataset (rare -- once per desired dataset size/quality bar)**

```
judge-eval generate-dataset --n-samples 50
```

Samples `--n-samples` gold rows from `data/original/xsum/test.parquet`, builds a clean example plus a verified summary-corruption and source-corruption for each, and **registers** the result as a named `mlflow.genai` evaluation dataset (`faithfulness-examples`) -- not just a local file. This is the expensive step (~4 LLM calls per gold row: 2 corruption generations + 2 referee verifications) and is meant to be run once and reused by everything else; `run-evaluation` loads the registered dataset directly rather than regenerating or re-deriving its shape on every invocation, since generation is rare and evaluation is frequent.

Each generated row is split into `expectations` (`expected_faithful`, `corruption_description` -- the two fields `FaithfulnessEvaluator` actually reads to grade the judge) and `tags` (`corruption_target`, `corruption_type`, `held_out`, `example_id` -- metadata used only for splitting/grouping, never consulted by the scoring logic). The run logs its params (`n_samples`, model tiers, held-out ratio), its summary metrics (clean/corrupted/dropped/held-out counts), and the registered dataset's id/digest. A copy is still written to `data/generated/examples.parquet` for quick manual inspection, but it's no longer what `run-evaluation` reads.

The sample is deterministic (`random_state=0`), so re-running with the same `--n-samples` reproduces the same dataset. Note: increasing `--n-samples` does *not* append to the existing dataset -- it re-samples from scratch and replaces the registered `faithfulness-examples` dataset (old records are deleted before the new ones are merged in). There's no incremental "grow the dataset" path yet.

Each gold row is deterministically assigned to the `opt` split or the `eval` (held-out) split (~20%, by gold id) and this is stamped per-example as the `held_out` tag.

**2. Evaluate a judge against the dataset (frequent -- once per candidate judge/prompt/model)**

```
judge-eval run-evaluation --judge custom --judge-model openai:/gpt-5.4-mini-2026-03-17 --split opt
```

- `--judge`: `custom` (our `make_judge`-based judge in `src/judge_eval/judges/faithfulness.py`) or `is_grounded` (MLflow's builtin groundedness judge, useful as a free baseline).
- `--judge-model`: any `<provider>:/<model>` string, for either judge backend.
- `--split`: `opt` while iterating on the judge/prompt, `eval` for a final, untouched-until-you-mean-it accuracy read, `all` to run both.

Each invocation is one MLflow run in the `judge-eval-judge-evaluation` experiment (named `{judge}_{model}_{split}`), scored via `mlflow.genai.evaluate` against `src/judge_eval/harness/evaluator.py`'s `FaithfulnessEvaluator` -- a judge-agnostic scorer that implements the four cases below and never needs to know how the judge under test was built. `mlflow.genai.evaluate` parallelizes the judge calls and per-example scoring, and every example's judge output + verdict + rationale is attached to that example's trace, browsable in the MLflow UI's Traces tab -- no more hand-diffing parquet files across runs. The run logs overall accuracy and a human-review-flag count as metrics; the per-`(corruption_target, corruption_type)` breakdown prints to stdout when available (it depends on every example's trace round-tripping through the tracking backend, so treat it as best-effort).

**3. Optimize the judge**

MLflow's `Judge.align(traces, optimizer)` is the intended mechanism for this stage (confirmed available in `mlflow==3.16.1`). Its prerequisite -- traces with feedback attached -- is now met by `run-evaluation`'s `mlflow.genai.evaluate` call, so wiring up an alignment/GEPA optimization pass against a `judge-eval-judge-evaluation` run's traces is the next piece to build; the orchestration and scoring side is done.