Plan:

- Generate a dataset by supplementing gold (source, summary) examples with corrupted ones, using an LLM instructed to corrupt the example summaries in specific ways.  In this way we produce labels of "true" / "false", corresponding to whether the data was faithful (not corrupted), along with metadata including a specific perterbation and the category of perterbation.  The former is useful as rationale for a judge evaluator, the latter for grouping runs to identify trends in judge behavior (missing specific categories more)

  - The corruption generator model should be a different (and ideally stronger) model family than whatever we land on as the judge, so the two don't share blind spots -- otherwise "hard" cases risk being hard only in ways the judge model already handles well.

  - Corruptions are not currently verified beyond "non-empty" and "not a no-op" (see `dataset/validate.py`). Since the judge evaluator trusts these labels as ground truth, add a verification step -- e.g. a second LLM check (or human spot-check of a sample) confirming the corruption actually broke faithfulness as instructed, and that it was a minimal single-fact change -- before the dataset is used for evaluation. Not yet implemented.

  - The current dev sample size (20 rows in `generate_dataset.py`) is a scaffold only; once corruption type x target is factored in, a meaningfully larger sample will be needed to get trustworthy per-category accuracy numbers.

- Evaluate the judge by feeding the dataset through the judge to produce output, that output being in turn judged by a Judge Evaluator.  The Judge Evaluator's task is to identify whether the Judge succeeded or failed at identifying unfaithful examples.  The task is made simple by the labeled data produced by the data generation process; there are four possible states that the evaluator will need to consider when comparing the Judge's verdict to the label:
  1. Judge value: Yes, Label: No -- the judge has missed at least one example of corruption; the evaluator judges this a failure automatically and leverages the description of the induced corruption as rationale. (We assume that the data generation process succeeded at its task.  We accept the risk that the induced corruption has masked another corruption on the basis that even a single missed example is problematic) Verdict: Falure
  2. Judge value: No, Label: Yes -- the judge claims to have found an example of corruption that was not induced; this is possibly spurious and possibly a reflection of a natural corruption in the gold data -- example is preliminarily marked as a judge failure but flagged for human review
  3. Judge value: Yes, Label: Yes -- the judge sees no hallucinations in an uncorrupted example; we assume that this is accurate, verdict: Success
  4. Judge value: No, Label: No -- the judge identifies a corruption in a file which had an induced corruption.  It's possible that the judge is identifying some other feature of the example and is thus "right for the wrong reason" -- or perhaps even "right for the right reason, just not the reason we expected"; an LLM is invoked to compare the rationales for each -- if the rationales match, verdict is success, if the rationales differ, mark as failure and flag for human review.


- Before entering an optimization loop, hold out a fixed evaluation split that is never touched during prompt optimization. Otherwise optimizing the judge's prompt against the same dataset used to score it risks overfitting to artifacts of our own corruption generator rather than improving real hallucination detection.

- If the initial evaluation shows an insufficiently capable Judge or one that is prohibitive on cost or resources, enter an optimization process.  To increase scores, conduct a prompt optimization (using the same scoring mechanism, applied only to the non-held-out split), and/or move to a costly model and optimize. To decrease cost, move to a less costly model, evaluate, and optimize prompt as necessary.

## Usage

Two scripts, run at very different frequencies:

**1. Generate the dataset (rare -- once per desired dataset size/quality bar)**

```
python generate_dataset.py --n-samples 50
```

Samples `--n-samples` gold rows from `dataset/original/xsum/test.parquet`, builds a clean example plus a verified summary-corruption and source-corruption for each, and writes all of them to `dataset/generated/examples.parquet`. This is the expensive step (~4 LLM calls per gold row: 2 corruption generations + 2 referee verifications) and is meant to be run once and then reused by everything else -- `run_evaluation.py` never regenerates it.

The sample is deterministic (`random_state=0`), so re-running with the same `--n-samples` reproduces the same dataset. Note: increasing `--n-samples` does *not* append to the existing file -- it re-samples from scratch and overwrites `examples.parquet`. There's no incremental "grow the dataset" path yet.

Each gold row is deterministically assigned to the `opt` split or the `eval` (held-out) split (~20%, by gold id) and this is stamped per-example as `held_out`.

**2. Evaluate a judge against the dataset (frequent -- once per candidate judge/prompt/model)**

```
python run_evaluation.py --judge custom --judge-model openai:/gpt-5.4-mini-2026-03-17 --split opt
```

- `--judge`: `custom` (our `make_judge`-based judge in `faithfulness_judge.py`) or `is_grounded` (MLflow's builtin groundedness judge, useful as a free baseline).
- `--judge-model`: any `<provider>:/<model>` string, for either judge backend.
- `--split`: `opt` while iterating on the judge/prompt, `eval` for a final, untouched-until-you-mean-it accuracy read, `all` to run both.

Every invocation writes a **new** file to `dataset/generated/results/`, named `{judge}_{model}_{split}_{timestamp}.parquet` -- results from different runs are never clobbered, so you can line up multiple candidate judges/prompts/models side by side. Prints overall accuracy, a human-review-flag count, and per-`(corruption_target, corruption_type)` accuracy.

**3. Optimize the judge (not yet implemented)**

MLflow's `Judge.align(traces, optimizer)` is the intended mechanism (confirmed available in `mlflow==3.16.1`), but it consumes MLflow `Trace` objects with feedback attached -- nothing in `run_evaluation.py` currently produces traces, since judges are called directly (`judge(inputs=..., outputs=...)`) outside any tracing/run context. Wiring this up (likely via `mlflow.genai.evaluate` or manual `@mlflow.trace`) is a prerequisite for this stage, not yet built.

**Known ergonomic gaps:** dataset generation has no append/versioning (see above), and results are flat per-run parquet files rather than MLflow-tracked experiment runs -- fine for a handful of comparisons, but if you start sweeping many judge/model/prompt combinations, moving `run_evaluation.py` onto `mlflow.genai.evaluate` (or at least `mlflow.start_run()`) would get you comparison/search in the MLflow UI instead of hand-diffing parquet files.