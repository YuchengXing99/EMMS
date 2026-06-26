"""
5-Fold Cross-Validation for WSI + RNA Dual-Modal Survival Prediction
NO VALIDATION SET VERSION (Patient-Level)

All parameters are read from configs/default_config.py

Usage:
    python scripts/run.py --cancer_type BLCA
    python scripts/run.py  # Run all 6 cancer types
"""
import os
# Force single thread for reproducibility
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import argparse
import numpy as np
import pandas as pd
import torch
import random
from evidential_survival import ENNreg_init, ENN_survival_prediction
from evidential_survival import Loss_function, Eval_Loss_function, evreg_evaluation
from data_loader import DataPreprocessor, MultiModalDataset, get_collate_fn
from trainer import EVREGTrainer
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
from default_config import get_wsi_rna_config, get_wsi_rna_with_missing_config


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # dict ordering
    random.seed(seed)                          # Python RNG
    np.random.seed(seed)                       # NumPy RNG (data shuffle)
    torch.manual_seed(seed)                    # PyTorch RNG (weight init)
    torch.set_default_dtype(torch.float64)


def run_single_cancer(cancer_type, args):
    """Run 5-fold CV for a single cancer type"""
    if args.missing_config_train:
        config = get_wsi_rna_with_missing_config(args.missing_config_train)
    else:
        config = get_wsi_rna_config()
    set_seed(config.data.seed)

    # Auto-generate output directory based on actual parameters
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Pick the output dir depending on whether a missing-modality config is set
        if config.missing.enabled and config.missing.missing_config_train:
            # Build a suffix like "W30_R30" from "WSI:0.3_RNA:0.3"
            miss_suffix = config.missing.missing_config_train.replace(":", "").replace("WSI", "W").replace("RNA", "R").replace("0.", "")
            output_dir = os.path.join(
                script_dir, '..', 'results',
                f'missing_{miss_suffix}',
                cancer_type
            )
        else:
            output_dir = os.path.join(
                script_dir, '..', 'results',
                'missing_modality_W0_R0',
                cancer_type
            )
    else:
        output_dir = args.output_dir
    print(f"Output directory: {output_dir}")

    # Create output directory and clean old results
    os.makedirs(output_dir, exist_ok=True)

    old_detailed = os.path.join(output_dir, 'detailed_results.csv')
    old_summary = os.path.join(output_dir, 'summary_results.csv')
    if os.path.exists(old_detailed):
        os.remove(old_detailed)
        print(f"Removed old detailed_results.csv")
    if os.path.exists(old_summary):
        os.remove(old_summary)
        print(f"Removed old summary_results.csv")

    for k in range(5):
        old_model = os.path.join(output_dir, f'best_model_k{k}.pth')
        if os.path.exists(old_model):
            os.remove(old_model)
            print(f"Removed old best_model_k{k}.pth")

    # Initialize data preprocessor
    preprocessor = DataPreprocessor(config)
    preprocessor.load_wsi_embeddings(config.data.titan_embeddings_dir)

    all_results = []

    print("=" * 80)
    print(f"5-Fold Cross-Validation: WSI + RNA (NO VALIDATION SET, Patient-Level)")
    print("=" * 80)
    print(f"Cancer: {cancer_type}, K={config.model.K}")
    print(f"Learning rate={config.training.learning_rate}, Batch size={config.training.batch_size}")
    print(f"Max epochs={config.training.max_epochs} (fixed, no early stopping)")
    print(f"RNA gamma scale: {args.rna_gamma_scale}")
    print(f"align_weight (KL): {args.align_weight}")
    if args.rna_discount is not None or args.wsi_discount is not None:
        print(f"Custom discount: RNA={args.rna_discount}, WSI={args.wsi_discount}")
    if config.missing.enabled:
        print(f"Missing config (DisPro-style): {config.missing.missing_config_train}")
        print(f"  - complete_cases_only: {config.missing.complete_cases_only}")
        print(f"  - disjoint: {config.missing.disjoint}")
    print("=" * 80)

    # 5-Fold loop
    for k_fold in range(5):
        print(f"\n{'=' * 80}")
        print(f"Fold {k_fold}/4")
        print(f"{'=' * 80}")

        # Reset the same seed each fold so fold differences come only from the data split
        set_seed(config.data.seed)

        # Data paths
        rna_dir = os.path.join(config.data.mmp_root, 'data_csvs', 'rna', 'hallmarks', cancer_type)
        split_dir = os.path.join(
            config.data.mmp_root, 'splits', 'survival',
            f'TCGA_{cancer_type}_overall_survival_k={k_fold}'
        )

        rna_data = preprocessor.load_rna_data(rna_dir)
        train_split = pd.read_csv(os.path.join(split_dir, 'train.csv'))
        test_split = pd.read_csv(os.path.join(split_dir, 'test.csv'))

        # Unified data preparation (NO validation split, patient-level)
        result = preprocessor.prepare_data(
            train_split, test_split,
            modalities=['RNA', 'WSI'],
            rna_data=rna_data,
            with_validation=False,
            random_state=1,
            # missing modality simulation (only on train split)
            missing_config_train=config.missing.missing_config_train if config.missing.enabled else None,
            missing_seed=config.missing.missing_seed,
            missing_complete_cases_only=config.missing.complete_cases_only,
            missing_disjoint=config.missing.disjoint,
            missing_verbose=config.missing.verbose,
        )
        train_data = result['train']
        test_data = result['test']
        rna_cols = result['rna_cols']

        print(f"Train: {len(train_data)} patients (ALL used, no val split), Test: {len(test_data)} patients")

        modalities = ['RNA', 'WSI']

        # Extract features (patient-level)
        features_tr, masks_tr, dur_tr, evt_tr = preprocessor.extract_features(
            train_data, modalities, rna_cols
        )
        features_te, masks_te, dur_te, evt_te = preprocessor.extract_features(
            test_data, modalities, rna_cols
        )

        # Standardization (masks handle missing data)
        features_tr, dur_tr_transformed, idx_tr = preprocessor.fit_scalers(
            features_tr, dur_tr, masks=masks_tr
        )
        dur_tr_orig = dur_tr[idx_tr]
        evt_tr = evt_tr[idx_tr]
        masks_tr = {k: v[idx_tr] for k, v in masks_tr.items()}

        features_te, dur_te_transformed, idx_te = preprocessor.transform_data(
            features_te, dur_te, masks=masks_te
        )
        dur_te_orig = dur_te[idx_te]
        evt_te = evt_te[idx_te]
        masks_te = {k: v[idx_te] for k, v in masks_te.items()}

        print(f"Event rate - Train: {evt_tr.mean():.2%}, Test: {evt_te.mean():.2%}")

        # Create dataset using unified dict format
        train_dataset = MultiModalDataset(
            features_tr, dur_tr_transformed, evt_tr, masks=masks_tr
        )

        collate_fn = get_collate_fn(modalities, with_mask=True)
        train_loader = DataLoader(
            train_dataset, batch_size=config.training.batch_size,
            shuffle=True, collate_fn=collate_fn,
            num_workers=0
        )

        # Model initialization using unified dict format
        x_train_dict_init = {k: torch.tensor(v, dtype=torch.float64) for k, v in features_tr.items()}
        mask_dict_init = {k: torch.tensor(v, dtype=torch.float64) for k, v in masks_tr.items()}
        dur_tr_init = torch.tensor(dur_tr_transformed, dtype=torch.float64)

        print(f"\nInitializing model with masks:")
        print(f"  RNA: {mask_dict_init['RNA'].sum():.0f}/{len(mask_dict_init['RNA'])} valid")
        print(f"  WSI: {mask_dict_init['WSI'].sum():.0f}/{len(mask_dict_init['WSI'])} valid")

        prototypes, k_dict = ENNreg_init(x_train_dict_init, dur_tr_init, K_dict=config.model.K, mask_dict=mask_dict_init, rna_gamma_scale=args.rna_gamma_scale)

        # hx compression mode + discount init values
        discount_init = {}
        if args.rna_discount is not None:
            discount_init['RNA'] = args.rna_discount
        if args.wsi_discount is not None:
            discount_init['WSI'] = args.wsi_discount

        if discount_init:
            print(f"Custom discount_init: {discount_init}")

        model = ENN_survival_prediction(
            x_train_dict_init, k_dict, prototypes,
            discount_init=discount_init if discount_init else None
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.training.milestones, gamma=config.training.gamma
        )

        criterion = Loss_function(align_weight=args.align_weight)
        criterion_eval = Eval_Loss_function()

        trainer = EVREGTrainer(model, optimizer, scheduler, criterion, criterion_eval, config)
        save_path = os.path.join(output_dir, f'best_model_k{k_fold}.pth')
        # print_hx_every: print hx stats every N epochs (None to disable)
        trainer.train_no_val(train_loader, save_path, use_mask=True, print_every=1, max_epochs=config.training.max_epochs, print_hx_every=1)

        # Evaluation
        print(f"\n{'=' * 80}")
        print("Evaluation on Test Set...")
        print(f"{'=' * 80}")

        try:
            model.load_state_dict(torch.load(save_path, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(save_path))
        model.eval()

        x_test_dict = {k: torch.tensor(v, dtype=torch.float64) for k, v in features_te.items()}
        n_test = len(features_te['RNA'])
        durations_test_t = torch.tensor(dur_te_orig, dtype=torch.float64)

        # Three test scenarios (matches the paper: Genomics-only / Pathology-only / Complete)
        test_scenarios = {
            'RNA_only': {
                'RNA': torch.ones(n_test, dtype=torch.float64),
                'WSI': torch.zeros(n_test, dtype=torch.float64)
            },
            'WSI_only': {
                'RNA': torch.zeros(n_test, dtype=torch.float64),
                'WSI': torch.ones(n_test, dtype=torch.float64)
            },
            'Complete': {
                'RNA': torch.ones(n_test, dtype=torch.float64),
                'WSI': torch.ones(n_test, dtype=torch.float64)
            }
        }

        for scenario_name, masks_test in test_scenarios.items():
            print(f"\n  Evaluating: {scenario_name}")
            with torch.no_grad():
                pred = model(x_test_dict, masks=masks_test)

            for lam in config.eval.eval_lambdas:
                ci, ibs, nbll = evreg_evaluation(
                    pred, durations_test_t, evt_te,
                    weight=lam, pt=preprocessor.pt, YJ=True,
                    durations_train=dur_tr_orig, events_train=evt_tr, ibs_bins=4
                )
                all_results.append({
                    'fold': k_fold,
                    'test_scenario': scenario_name,
                    'lambda': lam,
                    'C-index': ci,
                    'IBS': ibs,
                    'NBLL': nbll
                })

            scenario_results = [r for r in all_results
                                if r['fold'] == k_fold and r['test_scenario'] == scenario_name]
            best_ci = max(r['C-index'] for r in scenario_results)
            print(f"    {scenario_name} best C-index: {best_ci:.4f}")

    # Summary
    print(f"\n{'=' * 80}")
    print("5-Fold Cross-Validation Summary (WSI + RNA, NO VALIDATION)")
    print(f"{'=' * 80}")

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)

    # Aggregate by (test_scenario, lambda)
    summary_rows = []
    for scenario in ['RNA_only', 'WSI_only', 'Complete']:
        print(f"\n  Test Scenario: {scenario}")
        for lam in config.eval.eval_lambdas:
            subset = df_all[(df_all['test_scenario'] == scenario) & (df_all['lambda'] == lam)]
            summary_rows.append({
                'test_scenario': scenario,
                'lambda': lam,
                'C-index_mean': subset['C-index'].mean(),
                'C-index_std': subset['C-index'].std(),
                'IBS_mean': subset['IBS'].mean(),
                'IBS_std': subset['IBS'].std(),
                'NBLL_mean': subset['NBLL'].mean(),
                'NBLL_std': subset['NBLL'].std()
            })

        scenario_summary = [r for r in summary_rows if r['test_scenario'] == scenario]
        best_row = max(scenario_summary, key=lambda x: x['C-index_mean'])
        print(f"    Best Lambda={best_row['lambda']:.1f}: "
              f"C-index={best_row['C-index_mean']:.4f}+/-{best_row['C-index_std']:.4f} | "
              f"IBS={best_row['IBS_mean']:.4f} | NBLL={best_row['NBLL_mean']:.4f}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(output_dir, 'summary_results.csv'), index=False)
    print(f"\n{'=' * 80}")


def main(args):
    """Run experiments for one or all cancer types"""
    # The four cancer cohorts reported in the paper
    ALL_CANCERS = ['BRCA', 'LUAD', 'STAD', 'KIRC']

    if args.cancer_type is None:
        n = len(ALL_CANCERS)
        print("=" * 80)
        print(f"Running all {n} cancer types with best configuration")
        print("=" * 80)
        for i, cancer in enumerate(ALL_CANCERS, 1):
            print(f"\n{'#' * 80}")
            print(f"# [{i}/{n}] Processing {cancer}")
            print(f"{'#' * 80}\n")
            run_single_cancer(cancer, args)
        print("\n" + "=" * 80)
        print("All cancer types completed!")
        print("=" * 80)
    else:
        run_single_cancer(args.cancer_type, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='5-Fold CV for WSI+RNA (NO validation set)')
    parser.add_argument('--cancer_type', type=str, default=None,
                        choices=['BRCA', 'LUAD', 'STAD', 'KIRC'],
                        help='Cancer type (default: run all 4 cancers)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results and models (default: auto-generated based on cancer type)')
    parser.add_argument('--rna_gamma_scale', type=float, default=0.3,
                        help='RNA gamma scaling factor for L2 normalization (default: 0.3)')
    # Discount init values (control fusion weights)
    parser.add_argument('--rna_discount', type=float, default=None,
                        help='RNA discount init value (default: 0.0)')
    parser.add_argument('--wsi_discount', type=float, default=None,
                        help='WSI discount init value (default: 0.0)')
    # KL alignment loss weight
    parser.add_argument('--align_weight', type=float, default=0.01,
                        help='Weight for complete-case Gaussian KL alignment loss (default: 0.01; set 0.0 to disable)')

    # Artificial missing modality
    parser.add_argument('--missing_config_train', type=str, default=None,
                        help='missing config, e.g., "WSI:0.3_RNA:0.3". '
                             'Available presets: WSI:0.0_RNA:0.6, WSI:0.2_RNA:0.4, '
                             'WSI:0.3_RNA:0.3, WSI:0.4_RNA:0.2, WSI:0.6_RNA:0.0')

    args = parser.parse_args()
    main(args)
