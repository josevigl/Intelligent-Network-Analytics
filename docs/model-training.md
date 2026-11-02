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
