# Model training and comparison

## Methodology

`ina-train-models` loads only the source `train` split and creates local train,
validation, and test partitions. TelecomTS has no explicit recording or
scenario-ID column. It does contain `start_time` and `end_time`; samples whose
time intervals overlap are assigned to the same group. Those groups are kept
intact in the outer partitions and cross-validation folds. If those time
columns are unavailable, the splitter falls back to stratified sample-level
splits and records that choice in the comparison results.

Preprocessing is fitted independently inside each CV fold. Candidate families
are ranked by mean CV macro F1, then their best parameter sets are fitted on
the outer training partition and compared on validation. Binary,
temporal-localization, and multi-label thresholds are calibrated on validation.
Only the selected model is evaluated against test; the test set does not affect
model choice, hyperparameters, or thresholds.

Affected-KPI labels can be constant within a grouped CV fold (for example,
labels present in every training window). The One-vs-Rest and Classifier Chain
logistic models use a constant binary predictor for those outputs instead of
failing when logistic regression sees only one class.

This is classical machine-learning training: there is no neural-network
checkpoint. The selected checkpoint-equivalent is the fitted estimator,
preprocessor, parameters, and validation-calibrated threshold saved together.

## Files

- `model_config.yaml`: CV settings, threshold policy, candidate algorithms, and
  their parameter grids for each objective.
- `config.yaml`: source feature selection, sequence length, train/validation/
  test ratios, and preprocessing settings.
- `src/preprocessing/data.py`: task targets and overlap-based timestamp groups.
- `src/preprocessing/splitting.py`: disjoint, group-aware outer partitions.
- `src/models/experiments.py`: fold-local preprocessing, CV search, validation
  comparison, threshold selection, final test evaluation, and artifact saving.
- `pipelines/train_models.py`: end-to-end CLI and comparison CSV/JSON writer.
- `pipelines/mlrun_components.py`, `pipelines/mlrun_workflow.py`, and
  `pipelines/mlrun_registry.py`: MLRun registration, inference, and evaluation.
- `src/artifacts.py` and `src/inference.py`: model bundle persistence and
  inference without refitting.
- `notebooks/ModelComparison.ipynb`: per-objective CV/validation charts and the
  selected model's final test score.

## Install and execute

From the repository root, activate the project's Python environment and install
the dataset/model extras:

```powershell
.\myenv\Scripts\Activate.ps1
python -m pip install -e ".[data,models]"
```

To run the comparison notebook too, install its optional visualization tools:

```powershell
python -m pip install -e ".[data,models,notebooks]"
```

Train and compare all four objectives:

```powershell
ina-train-models
```

Train one objective, while retaining the default project and model configs:

```powershell
ina-train-models --task binary_detection
ina-train-models --task anomaly_type
ina-train-models --task temporal_localization
ina-train-models --task affected_kpis
```

Multiple `--task` options can be combined. To use custom configuration or
output paths:

```powershell
ina-train-models `
  --config config.yaml `
  --models-config model_config.yaml `
  --output results/models
```

The command defaults to the Hugging Face `train` split. Keep `--split train` for
model development; the command constructs its own held-out partitions. Results
and one artifact bundle per model family are written under
`results/models/<task>/`. `results/models/comparison.csv` is the notebook input.

Run inference from any saved model-family artifact:

```powershell
ina-infer `
  --artifact results/models/binary_detection/random_forest `
  --split train `
  --output results/predictions/binary_detection.json
```

The public TelecomTS repository currently exposes a `train` split only; the
internal test partition is created during model training and is evaluated
there. The example uses the available source split as an inference batch.

Open `notebooks/ModelComparison.ipynb` after training to compare CV and
validation scores. Test scores are populated only for the selected model of
each objective.

## MLRun registry, inference, and evaluation

Training and model selection remain local with `ina-train-models`. MLRun is
used after training for model registration/versioning, run provenance, batch
inference, and evaluation on the exact held-out partition used by training.
Online deployment and monitoring are separate follow-up stages.

Install the optional MLRun dependencies in a Linux or WSL2 environment that
can access the MLRun API and artifact store. The MLRun extra includes S3
support because MLRun stores registered model artifacts in the configured
object store:

```text
python -m pip install -e ".[mlrun,data,models]"
```

First train locally. The default runs all four objectives; individual or
multiple objectives can be selected as above. Then configure the two
post-training workflows once. The source must point at a pushed revision
containing the MLOps code; do not put credentials in the URL. The inference
and evaluation functions request 1 GiB of memory and have a 4 GiB limit:

```text
ina-train-models
ina-mlrun-setup --source "git://github.com/<owner>/<repository>.git#<branch>"
```

If a function is already registered, rerun `ina-mlrun-setup` after pushing
this configuration change, then resubmit the failed workflow. The cluster
must have enough allocatable memory on a node for the 1 GiB request; if it
does not, use smaller configured resources or a larger-memory node.

Register each selected local model bundle and its `comparison.json` metadata
with MLRun. By default, all four objectives are registered from
`results/models`; repeat `--task` to limit registration:

```text
ina-mlrun-register --source "git://github.com/<owner>/<repository>.git#<branch>"
ina-mlrun-register --source "git://github.com/<owner>/<repository>.git#<branch>" --task anomaly_type
```

Registration only accepts the candidate marked selected by the local
comparison results. The model artifact includes its fitted preprocessing and
target transformations. MLRun also stores the comparison, CV/validation
metrics, source dataset and split, code revision, and a holdout manifest. The
manifest identifies the exact test positions and fingerprints the dataset and
configuration; evaluation fails if these no longer match.

In the MLRun UI, copy the registered model URI and the corresponding
`<task>_holdout_manifest` artifact URI. Use the model URI to run batch
inference on the available source `train` split:

```text
ina-mlrun-infer --model-uri "<registered-model-uri>"
```

The inference component resolves this registry URI to the stored `.joblib`
bundle through MLRun's model-artifact API before loading it. The launcher
synchronizes registered functions before submitting the workflow so new
function resources are used by newly submitted runs.

Evaluate the registered model on its recorded held-out test partition:

```text
ina-mlrun-evaluate \
  --model-uri "<registered-model-uri>" \
  --holdout-manifest-uri "<registered-holdout-manifest-uri>"
```

The evaluation command defaults to the same `AliMaatouk/TelecomTS` train split
and `config.yaml`; provide matching `--dataset`, `--split`, or `--config` values
if those differed at registration. Do not retrain between registration and
evaluation or change the data/configuration: the manifest check intentionally
rejects mismatches. Configure a serving runtime and monitoring stack
separately when the registry, inference, and evaluation flow has been verified.
Inference and evaluation load the dataset in their respective MLRun function;
the full DataFrame is not serialized as an intermediate pipeline parameter.
