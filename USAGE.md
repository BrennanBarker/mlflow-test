# Usage

A complete walkthrough of generating a dataset and evaluating a judge against it. See [README.md](README.md) for the design rationale and package layout.

## Prerequisites

- Python >= 3.11.
- `OPENAI_API_KEY` set in the environment (and `OPENAI_BASE_URL` too, if you're pointing the raw `openai` client at an enterprise gateway rather than api.openai.com -- everything here goes through `openai.OpenAI()`, which reads both from the environment).
- Gold (document, summary) data at `data/original/xsum/test.parquet` (columns: `id`, `document`, `summary`).

## Install

```
pip install -e .
```

Installs the `judge-eval` CLI (`src/judge_eval/cli.py:main`, per `pyproject.toml`'s `[project.scripts]`) in editable mode, so code changes take effect immediately without reinstalling.

## Tracking

Every `judge-eval` invocation is wrapped in an MLflow run, in whichever tracking store `mlflow.get_tracking_uri()` resolves to (a local file or sqlite store by default in most environments -- run `mlflow ui` from the same directory to browse it).

- Dataset generation runs land in the `judge-eval-dataset-generation` experiment.
- Judge evaluation runs land in `judge-eval-judge-evaluation`.

They're kept as separate experiments because they run at very different frequencies (generation is rare, evaluation is frequent) -- you'll usually only want to browse one at a time, and this keeps the UI's run list from mixing them.

`mlflow.openai.autolog()` and `mlflow.litellm.autolog()` are both enabled at CLI startup (`common/tracking.py::enable_autologging`, called from `cli.py::main`), so every LLM call made by either pipeline shows up as a trace on its run:

- **openai.autolog** covers the "oracle" calls -- corruption generation, corruption verification, and the harness's rationale comparison -- all of which go through a raw `openai.OpenAI()` client (`common/structured_call.py`).
- **litellm.autolog** covers the judge-under-test's own calls: `make_judge` (model strings like `openai:/...`) and the builtin `is_grounded` judge both route through MLflow's litellm adapter internally, not the raw openai client.

## Step 1: Generate the dataset

```
judge-eval generate-dataset --n-samples 50
```

What this does:

1. Samples `--n-samples` gold rows (deterministically, `random_state=0`) from `data/original/xsum/test.parquet`.
2. For each row, generates a clean example plus a verified summary-corruption and source-corruption (`dataset/augment.py`) -- up to 4 LLM calls per gold row: 2 corruption generations (`CORRUPTION_MODEL`) and 2 corruption verifications (`REFEREE_MODEL`). Corruptions that fail verification are dropped rather than kept with a bad label.
3. Deterministically assigns each gold row to the `opt` split or the held-out `eval` split (~20%, by gold row id).
4. Registers the resulting examples as a named MLflow evaluation dataset, `faithfulness-examples` (`mlflow.genai.create_dataset` + `merge_records`), inside a run in the `judge-eval-dataset-generation` experiment. Also writes a copy to `data/generated/examples.parquet` for quick manual inspection (`pd.read_parquet`), though nothing in `run-evaluation` reads that file -- it loads the registered dataset instead.

Each row is split into two parts when it's registered:

- **`expectations`**: `expected_faithful`, `corruption_description` -- the two fields `FaithfulnessEvaluator` (the scorer used in step 3) actually reads to grade a judge.
- **`tags`**: `corruption_target`, `corruption_type`, `held_out`, `example_id` -- metadata used only for splitting and the per-category breakdown, never consulted by the scoring logic itself. (Non-corrupted rows get `corruption_target`/`corruption_type` = `"clean"` rather than a null value, both because that reads better in the breakdown and because MLflow's expectation logging rejects a null value outright.)

The run logs:

- **Params**: `n_samples`, `held_out_modulus`, `corruption_model`, `referee_model`, `gold_path`, and (after registration) `dataset_id`/`dataset_digest`.
- **Metrics**: `n_examples`, `n_clean`, `n_corrupted`, `n_dropped`, `n_held_out`.

Console output has this shape (illustrative numbers -- not a transcript of an actual run):

```
Wrote 87 examples to data/generated/examples.parquet
Registered as mlflow dataset 'faithfulness-examples' (d-...)
  clean: 44, corrupted: 43, dropped by verification: 5
  held out: 18, available for optimization: 69
corruption_target  corruption_type
source             contradiction         11
                    ...
summary            factual_alteration    12
                    ...

MLflow run: <run_id> (experiment: judge-eval-dataset-generation)
```

**Regenerating overwrites, it doesn't append.** Re-running `generate-dataset` -- with the same `--n-samples` or a different one -- deletes the existing `faithfulness-examples` dataset and its records, then registers a fresh one. There's no incremental "grow the dataset" path; every generation run replaces the last one wholesale. (Past generation *runs* themselves aren't deleted -- only the registered dataset they point at is replaced -- so old runs' params/metrics remain in the MLflow UI even though the dataset they describe no longer exists.)

## Step 2: Look at what got generated

```
mlflow ui
```

Open the `judge-eval-dataset-generation` experiment to see the run's params/metrics. The registered dataset itself is separate from any one run -- `mlflow.genai.search_datasets()` or the UI's dataset view will show `faithfulness-examples` with its record count and digest, independent of which run created it.

To inspect it from Python:

```python
import mlflow.genai as genai

dataset = genai.get_dataset(name="faithfulness-examples")
df = dataset.to_df()
print(len(df), "records")
print(df.iloc[0][["inputs", "expectations", "tags"]])
```

## Step 3: Evaluate a judge against the dataset

```
judge-eval run-evaluation --judge custom --judge-model openai:/gpt-5.4-mini-2026-03-17 --split opt
```

- `--judge`: `custom` (the `make_judge`-based judge in `src/judge_eval/judges/faithfulness.py`) or `is_grounded` (MLflow's builtin groundedness judge, useful as a free baseline).
- `--judge-model`: any `<provider>:/<model>` string, for either judge backend.
- `--split`: `opt` while iterating on the judge/prompt, `eval` for a final, untouched-until-you-mean-it accuracy read, `all` to run both.

What this does:

1. Loads the registered `faithfulness-examples` dataset and filters it by the `held_out` tag according to `--split`.
2. Builds the selected judge backend, then runs `mlflow.genai.evaluate(data=..., predict_fn=..., scorers=[FaithfulnessEvaluator()])` inside a run in the `judge-eval-judge-evaluation` experiment. `evaluate` parallelizes the judge calls and scoring across the dataset (a `ThreadPoolExecutor` under the hood) and attaches each example's judge output, verdict, and rationale to that example's trace.
3. `FaithfulnessEvaluator` (`harness/evaluator.py`) implements the four cases below, comparing the judge's verdict to the label:

   1. Judge: faithful, Label: not faithful -- the judge missed an induced corruption. Automatic failure; rationale is the corruption description.
   2. Judge: not faithful, Label: faithful -- the judge flagged a clean example. Failure, flagged for human review (possibly spurious, possibly a real flaw in the gold data).
   3. Judge: faithful, Label: faithful -- correct pass. Success.
   4. Judge: not faithful, Label: not faithful -- the judge caught *a* corruption, but maybe not *the* corruption. A cheap model (`RATIONALE_COMPARISON_MODEL`) compares the judge's rationale against the corruption description; matching rationales are a success, mismatches are a failure flagged for human review. Results are cached (`functools.lru_cache` on `(corruption_description, judge_rationale, model)`) since repeated evaluation/optimization runs frequently re-produce an identical rationale for the same example.

Console output has this shape (illustrative numbers -- not a transcript of an actual run):

```
Judge: custom (openai:/gpt-5.4-mini-2026-03-17), split=opt
Overall accuracy: 84.06% (69 examples)
Flagged for human review: 4

corruption_target  corruption_type
clean              clean                 0.977273
source             contradiction         0.727273
                    ...
summary            factual_alteration    0.833333
                    ...

MLflow run: <run_id> (experiment: judge-eval-judge-evaluation)
```

The per-category breakdown depends on every example's trace round-tripping through the tracking backend; if that doesn't happen for some reason, the run still completes and logs `accuracy`/`needs_human_review_count` (sourced from `mlflow.genai.evaluate`'s in-memory metrics, not the trace-derived table), and the console prints a note instead of the breakdown.

## Step 4: Compare judges/models/prompts

Every invocation is a separate MLflow run, so run it again with a different `--judge`, `--judge-model`, or `--split` and compare runs side by side in the `judge-eval-judge-evaluation` experiment -- sort/filter by the logged `accuracy` metric, or open a run's Traces tab to read individual judge rationales and verdicts.

```
judge-eval run-evaluation --judge is_grounded --judge-model openai:/gpt-5.4-mini-2026-03-17 --split opt
```

## Model tiers

Defined in `common/models.py`:

| Constant | Used for | Default | Why |
|---|---|---|---|
| `CORRUPTION_MODEL` | generating ground-truth corruptions | `gpt-5.4` | Oracle role, rare (once per gold row per generation run) -- use the strongest model available. |
| `REFEREE_MODEL` | verifying corruptions are minimal/valid | `gpt-5.4` | Same oracle role as above. |
| `RATIONALE_COMPARISON_MODEL` | case-4 rationale comparison in `FaithfulnessEvaluator` | `gpt-5.4-mini-2026-03-17` | Fires on every corrupted example on every evaluation run, and far more under prompt optimization -- cost matters here in a way it doesn't for the rare oracle calls above. |
| `DEFAULT_JUDGE_MODEL` | the judge under test's default `--judge-model` | `openai:/gpt-5.4-mini-2026-03-17` | Just a starting point; override per invocation. |

## Status

**Working**: dataset generation (with verification and MLflow dataset registration), judge evaluation via `mlflow.genai.evaluate` against either judge backend, full MLflow tracking (experiments, params, metrics, traces, autologging).

**Not yet built**: prompt optimization. The plan is `mlflow.genai.optimize_prompts` with a `GepaPromptOptimizer`, reusing `FaithfulnessEvaluator` as the scorer (its signature already matches what `optimize_prompts` calls scorers with) and running only against the `opt` split. The concrete gap: the judge's instructions (`judges/faithfulness.py`'s `INSTRUCTIONS` constant) need to become a registered `mlflow.genai` prompt via `register_prompt`, and a new predict_fn is needed for optimization specifically -- `optimize_prompts` calls `predict_fn(inputs)` with the whole inputs dict as one positional argument, unlike `evaluate`'s keyword-unpacking convention.

**Known limitations**:
- Dataset generation is single-threaded (`dataset/generate.py` calls `augment_row` sequentially per gold row) -- fine at tens of rows, will be slow at hundreds+.
- No end-to-end real run of `generate-dataset` has been done yet in this codebase's current form -- only individual pieces have been exercised against live data. Worth a small real run (`--n-samples 2-3`) to sanity-check before scaling up.
