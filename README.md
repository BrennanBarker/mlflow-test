Plan:

- Generate a dataset by supplementing gold (source, summary) examples with corrupted ones, using an LLM instructed to corrupt the example summaries in specific ways.  In this way we produce labels of "true" / "false", corresponding to whether the data was faithful (not corrupted), along with metadata including a specific perterbation and the category of perterbation.  The former is useful as rationale for a judge evaluator, the latter for grouping runs to identify trends in judge behavior (missing specific categories more)

  - The corruption generator model should be a different (and ideally stronger) model family than whatever we land on as the judge, so the two don't share blind spots -- otherwise "hard" cases risk being hard only in ways the judge model already handles well.

  - Corruptions are verified by a second LLM check (`dataset/verify.py`, using `REFEREE_MODEL`) confirming the corruption actually broke faithfulness as instructed and was a minimal single-fact change, before the dataset is used for evaluation. Corruptions that fail verification are dropped rather than included with a bad label (see the `dropped` count `generate-dataset` prints).

  - The current dev sample size (`--n-samples` default in `judge-eval generate-dataset`) is a scaffold only; once corruption type x target is factored in, a meaningfully larger sample will be needed to get trustworthy per-category accuracy numbers.

- Evaluate the judge by feeding the dataset through the judge to produce output, that output being in turn judged by a Judge Evaluator.  The Judge Evaluator's task is to identify whether the Judge succeeded or failed at identifying unfaithful examples.  The task is made simple by the labeled data produced by the data generation process; there are four possible states that the evaluator will need to consider when comparing the Judge's verdict to the label:
  1. Judge value: Yes, Label: No -- the judge has missed at least one example of corruption; the evaluator judges this a failure automatically and leverages the description of the induced corruption as rationale. (We assume that the data generation process succeeded at its task.  We accept the risk that the induced corruption has masked another corruption on the basis that even a single missed example is problematic) Verdict: Falure
  2. Judge value: No, Label: Yes -- the judge claims to have found an example of corruption that was not induced; this is possibly spurious and possibly a reflection of a natural corruption in the gold data -- example is preliminarily marked as a judge failure but flagged for human review
  3. Judge value: Yes, Label: Yes -- the judge sees no hallucinations in an uncorrupted example; we assume that this is accurate, verdict: Success
  4. Judge value: No, Label: No -- the judge identifies a corruption in a file which had an induced corruption.  It's possible that the judge is identifying some other feature of the example and is thus "right for the wrong reason" -- or perhaps even "right for the right reason, just not the reason we expected"; an LLM is invoked to compare the rationales for each -- if the rationales match, verdict is success, if the rationales differ, mark as failure and flag for human review.


- Before entering an optimization loop, hold out a fixed evaluation split that is never touched during prompt optimization. Otherwise optimizing the judge's prompt against the same dataset used to score it risks overfitting to artifacts of our own corruption generator rather than improving real hallucination detection.

- If the initial evaluation shows an insufficiently capable Judge or one that is prohibitive on cost or resources, enter an optimization process via `mlflow.genai.optimize_prompts` with a `GepaPromptOptimizer` (using the same scoring mechanism -- `FaithfulnessEvaluator` -- applied only to the non-held-out `opt` split), and/or move to a costly model and optimize. To decrease cost, move to a less costly model, evaluate, and optimize prompt as necessary. Not yet implemented -- see "Status" in [USAGE.md](USAGE.md).

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

## Usage

See [USAGE.md](USAGE.md) for a complete walkthrough: install, generate a dataset, evaluate a judge, and inspect results in the MLflow UI.