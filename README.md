# UA-DEP SARS-CoV-2 Pipeline

Uncertainty-Aware Deep Ensemble Pipeline for SARS-CoV-2 lineage classification and cyber-exposure robustness analysis.

## What It Does

- Builds `processed/combined.fasta` and `processed/labels.csv` from local class FASTA folders.
- Cleans sequences (uppercase, `A/C/G/T/N` only), filters by min length and max `N` fraction.
- Extracts k-mer frequency features (`k=3` default).
- Standardizes k-mer features using train-split statistics before model training/inference.
- Creates stratified train/calibration/validation/test splits (70/10/10/10).
- Trains one baseline 1D CNN and a 10-member deep ensemble (different seeds + bootstrap train resamples per member).
- Calibrates each member with temperature scaling on calibration split.
- Produces ensemble predictions and uncertainty scores (default rejection metric: `ensemble_probability_variance`).
- Selects rejection threshold `tau` from validation coverage-accuracy analysis or ROC-based validation error detection.
- Evaluates clean test performance (accuracy, macro-F1, confusion matrix, NLL, ECE, Brier, plots).
- Runs cyber-exposure simulations (substitution, truncation, chimeric) and reports detection/rejection/accepted-accuracy.

## Dataset Layout

Expected default raw path:

`./sars_cov_2_10class_dataset/raw/<class_folder>/ncbi_dataset/data/genomic.fna`

Examples:

- `alpha`
- `beta`
- `gamma`
- `delta`
- `omicron_ba1`
- `omicron_ba2`
- `omicron_ba5`
- `xbb`
- `xbb_1_5`
- `jn_1`

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python scripts/run_ua_dep.py \
  --raw-dir sars_cov_2_3class_dataset \
  --processed-dir processed \
  --output-dir outputs \
  --k 3 \
  --epochs 30 \
  --ensemble-size 10 \
  --threshold-selection-method coverage_accuracy \
  --device cpu
```

Recommended for stronger separation between close lineages:

```bash
python scripts/run_ua_dep.py \
  --raw-dir sars_cov_2_3class_dataset \
  --processed-dir processed \
  --output-dir outputs \
  --k 3 \
  --epochs 30 \
  --ensemble-size 10 \
  --min-coverage 0.85 \
  --threshold-selection-method roc \
  --device cpu
```

## Outputs

- `processed/combined.fasta`
- `processed/labels.csv`
- `processed/splits.csv`
- `outputs/models/`
- `outputs/metrics/`
- `outputs/predictions/`
- `outputs/plots/`
- `outputs/cyber_simulation/`

## Notes

- Uses only local FASTA files. No live NCBI downloads.
- Defaults are CPU-friendly
