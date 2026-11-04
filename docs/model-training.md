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
  `pipelines/mlrun_train.py`: MLRun handlers, parallel workflow, and launcher.
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

## MLRun training workflow

`ina-mlrun-train` registers and runs a two-stage MLRun workflow:

1. Load and register the source `train` split once as a Parquet dataset.
2. Run one parallel branch per selected objective. Each branch performs the
   existing group-aware CV and validation model selection, then records its
   metrics, comparison JSON, candidate bundles, and selected model in MLRun.

The selected model is registered with its preprocessing pipeline, feature
order, configuration, and target encoder in the same joblib bundle. MLRun also
stores its selected parameters and CV/validation/test metrics in the model
record. Only the selected candidate is evaluated against test, and
`--skip-test` is available for exploratory runs.

MLRun is optional for local model training. Install the project and MLRun
dependencies in a Linux or WSL2 shell configured to access an MLRun API/DB,
workflow engine, image builder, and image registry:

```text
python -m pip install -e ".[mlrun,data,models]"
```

MLRun must be able to access the repository source and build or pull the
runtime image. Commit and push the code first, then provide a repository or
archive URL accessible to the MLRun environment. Configure private-repository
credentials in MLRun, never in the URL. The workflow writes the generated
`project.yaml` into the repository directory.
Use the same MLRun version as the target platform; the optional dependency
installs its KFP 1.8 workflow extra.

Run all four branches:

```text
ina-mlrun-train --source "git://github.com/<owner>/<repository>.git"
```

Run only anomaly-type classification and avoid using the holdout test set
during an exploratory run:

```text
ina-mlrun-train --source "git://github.com/<owner>/<repository>.git" --task anomaly_type --skip-test
```

Select multiple branches by repeating `--task`. The workflow defaults to
`AliMaatouk/TelecomTS`, source split `train`, and the repository's `config.yaml`
and `model_config.yaml`. Override the dataset, project name, custom config
files, or workflow image with `--dataset`, `--project`, `--config`,
`--models-config`, `--base-image`, and `--image`. The `--image` value must be
available to the MLRun runtime; omit it to use MLRun's generated project image.
Use `--no-watch` to submit the workflow and return without waiting.

Monitor each run in the MLRun UI. Each task execution includes the selected
model and its parameters, CV/validation/test metrics, the complete candidate
comparison JSON, and the model bundles. The loader execution records the
dataset artifact and its row/column counts. To deploy a registered model or add
serving and monitoring components, configure the MLRun serving runtime and
downstream infrastructure separately.
