# EMMS

WSI + RNA survival prediction that handles missing modalities without imputation.
Each modality produces survival evidence (Dempster-Shafer / Gaussian random fuzzy
numbers); a missing modality is treated as vacuous evidence (h = 0) and simply
drops out of the fusion.

## Install

    pip install -r requirements.txt

Python 3.10+. Main deps: torch, scikit-learn, scikit-survival, pycox, lifelines.

## Data

Everything is read from `data/`:

- `titan_embeddings/` - one `.pt` per patient (TITAN WSI embedding, 768-d), named `TCGA-XX-XXXX.pt`
- `data_csvs/rna/hallmarks/<CANCER>/rna_clean.csv` - gene expression
- `splits/survival/TCGA_<CANCER>_overall_survival_k=<0..4>/` - `train.csv`, `test.csv`

Cancers: BRCA, LUAD, STAD, KIRC.

## Run

All four cancers, 5-fold, no missing modality:

    python scripts/run.py

One cancer:

    python scripts/run.py --cancer_type KIRC

Drop modalities on the training split (paired samples, 60% total):

    python scripts/run.py --cancer_type KIRC --missing_config_train WSI:0.3_RNA:0.3

`--missing_config_train` presets: `WSI:0.0_RNA:0.6`, `WSI:0.2_RNA:0.4`,
`WSI:0.3_RNA:0.3`, `WSI:0.4_RNA:0.2`, `WSI:0.6_RNA:0.0`.

The script takes one missing config at a time, so loop over them in the shell.
All five missing configs for one cancer (PowerShell):

    foreach ($cfg in "WSI:0.0_RNA:0.6","WSI:0.2_RNA:0.4","WSI:0.3_RNA:0.3","WSI:0.4_RNA:0.2","WSI:0.6_RNA:0.0") {
        python scripts/run.py --cancer_type KIRC --missing_config_train $cfg
    }

All five missing configs for all four cancers (drop `--cancer_type`):

    foreach ($cfg in "WSI:0.0_RNA:0.6","WSI:0.2_RNA:0.4","WSI:0.3_RNA:0.3","WSI:0.4_RNA:0.2","WSI:0.6_RNA:0.0") {
        python scripts/run.py --missing_config_train $cfg
    }

Each config writes to its own folder (`results/missing_W0_R6/`, `missing_W2_R4`,
...), so the runs do not overwrite each other.

Other flags:

- `--align_weight` - KL alignment loss on complete cases (default 0.01; pass 0.0 to turn off)
- `--rna_gamma_scale` - RBF gamma scaling (default 0.3)
- `--output_dir` - override the output path

The rest (K=50, 70 epochs, lr 0.011, batch 256, seed 123) is in
`configs/default_config.py`.

## Output

Written under `results/`. No missing config goes to
`results/missing_modality_W0_R0/<CANCER>/`; a missing config encodes the rates in
the folder name, e.g. `WSI:0.3_RNA:0.3` -> `results/missing_W3_R3/<CANCER>/`.

Each folder has:

- `detailed_results.csv` - one row per fold / test scenario / lambda
- `summary_results.csv` - mean and std over the 5 folds
- `best_model_k0.pth` ... `best_model_k4.pth`

Every trained model is tested under three scenarios (RNA only, WSI only, both) and
over lambda in [0, 1]; each row reports C-index, IBS and NBLL.

## Notebook

`pipeline.ipynb` runs the same thing for a single cancer, which is easier to read
through one fold at a time. Edit the config cell (`CANCER`, `MISSING_CONFIG`,
`ALIGN_WEIGHT`) and run all cells. The shipped example is KIRC with
`WSI:0.3_RNA:0.3` missing on train, tested on the complete set.
