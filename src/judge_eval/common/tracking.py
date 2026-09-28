"""MLflow experiment names and autologging setup shared by both pipelines."""
import mlflow
import mlflow.litellm
import mlflow.openai

DATASET_GENERATION_EXPERIMENT = "judge-eval-dataset-generation"
JUDGE_EVALUATION_EXPERIMENT = "judge-eval-judge-evaluation"

# Name of the registered mlflow.genai evaluation dataset that `generate-dataset`
# (re)writes and `run-evaluation` loads. Registering it (rather than just reading
# data/generated/examples.parquet) is what lets run-evaluation -- the frequent
# command -- skip re-deriving the dataset shape on every invocation.
FAITHFULNESS_DATASET_NAME = "faithfulness-examples"


def enable_autologging() -> None:
    """Trace every LLM call made by either pipeline.

    mlflow.openai.autolog() covers the oracle calls (corruption generation,
    verification, rationale comparison), which go through a raw openai client
    that can be pointed at an enterprise openai-flavored gateway.
    mlflow.litellm.autolog() covers the judge-under-test's own calls: both
    make_judge (model="openai:/...") and is_grounded route through litellm
    rather than the raw openai client, so autolog(openai) alone misses them.
    """
    mlflow.openai.autolog()
    mlflow.litellm.autolog()
