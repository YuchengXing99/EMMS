import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, PowerTransformer
from functools import partial
import os
from typing import List, Dict, Optional, Tuple, Any


# Dataset takes features/durations/events (optional masks); ds[i] returns
# (per-modality features..., duration_i, event_i[, per-modality mask_i])
class MultiModalDataset(Dataset):
    """Multi-modal dataset supporting any combination of modalities."""

    # features['WSI']:(N,768), features['RNA']:(N,4168) -> self.features (torch.float64)
    def __init__(
        self,
        features: Dict[str, np.ndarray],
        durations: np.ndarray,
        events: np.ndarray,
        masks: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.modality_names = list(features.keys())
        self.features = {k: torch.tensor(v, dtype=torch.float64) for k, v in features.items()}
        self.durations = torch.tensor(durations, dtype=torch.float64)
        self.events = torch.tensor(events, dtype=torch.float64)
        self.masks = {k: torch.tensor(v, dtype=torch.float64) for k, v in masks.items()} if masks else None
        self.with_mask = masks is not None

    # patient-level: len = #patients; slide-level: len = #slides
    def __len__(self) -> int:
        return len(self.durations)

    def __getitem__(self, idx: int) -> Tuple:
        items = [self.features[k][idx] for k in self.modality_names]
        items.extend([self.durations[idx], self.events[idx]])
        if self.with_mask:
            items.extend([self.masks[k][idx] for k in self.modality_names])
        return tuple(items)

    # Return the modality order of this dataset (used to build collate_fn)
    def get_modality_names(self) -> List[str]:
        return self.modality_names


# batch=[(RNA0,WSI0,dur0,evt0,mask0...),(RNA1,WSI1,...)] ->
# inputs{'RNA':(B,4168),'WSI':(B,768)}, durations(B,), events(B,)[, masks]
def collate_fn_generic(batch: List[Tuple], modality_names: List[str], with_mask: bool = False) -> Tuple:
    """Generic collate_fn that works for any combination of modalities."""
    n_mod = len(modality_names)
    inputs = {name: torch.stack([item[i] for item in batch]) for i, name in enumerate(modality_names)}
    durations = torch.stack([item[n_mod] for item in batch])
    events = torch.stack([item[n_mod + 1] for item in batch])

    if with_mask:
        masks = {
            name: torch.stack([item[n_mod + 2 + i] for item in batch]) for i, name in enumerate(modality_names)
        }
        return inputs, durations, events, masks
    return inputs, durations, events


# get_collate_fn(['RNA','WSI'], with_mask=True) -> a collate_fn ready to pass to DataLoader
def get_collate_fn(modality_names: List[str], with_mask: bool = False):
    return partial(collate_fn_generic, modality_names=modality_names, with_mask=with_mask)


class DataPreprocessor:

    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.pt = PowerTransformer(method='yeo-johnson')
        self.wsi_embeddings = None  # {patient_id: embedding}
        self.wsi_dim = None
        self._rna_cols = None

    def load_wsi_embeddings(self, embeddings_dir: str) -> None:
        self.wsi_embeddings = {}
        self.wsi_dim = None

        pt_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.pt')]
        print(f"  Loading {len(pt_files)} patient-level WSI embeddings...")

        for pt_file in pt_files:
            patient_id = pt_file.replace('.pt', '')  # TCGA-XX-XXXX.pt -> TCGA-XX-XXXX
            embedding = torch.load(os.path.join(embeddings_dir, pt_file), weights_only=True)

            # Handle different tensor formats
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            if embedding.ndim > 1:
                embedding = embedding.squeeze()

            self.wsi_embeddings[patient_id] = embedding.astype('float64')

            if self.wsi_dim is None:
                self.wsi_dim = len(embedding)

        print(f"  Loaded {len(self.wsi_embeddings)} patients, embedding dim = {self.wsi_dim}")

    # Read rna_clean.csv and extract submitter_id (keep only tumor samples, sample_type==01)
    def load_rna_data(self, rna_dir: str) -> pd.DataFrame:
        """Load and preprocess RNA expression data."""
        rna_data = pd.read_csv(os.path.join(rna_dir, 'rna_clean.csv'))
        if 'Unnamed: 0' in rna_data.columns:
            rna_data = rna_data.drop(columns=['Unnamed: 0'])

        if 'sample' in rna_data.columns:
            rna_data['submitter_id'] = rna_data['sample'].str.rsplit('-', n=1).str[0]
            rna_data['sample_type'] = rna_data['sample'].str.split('-').str[-1]
            rna_data = rna_data[rna_data['sample_type'] == '01'].drop(columns=['sample_type'])
        else:
            rna_data['submitter_id'] = rna_data[rna_data.columns[0]].astype(str)

        rna_data['submitter_id'] = rna_data['submitter_id'].astype(str)
        return rna_data[~rna_data['submitter_id'].duplicated()]

    # Standardize the ID column: prefer submitter_id; otherwise derive it from case_id
    def _standardize_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'submitter_id' not in df.columns:
            if 'case_id' in df.columns:
                df['submitter_id'] = df['case_id'].astype(str)
            else:
                raise KeyError("Neither 'submitter_id' nor 'case_id' found")
        else:
            df['submitter_id'] = df['submitter_id'].astype(str)
        return df

    # Handle sex/gender column naming differences
    def _standardize_sex_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'sex' not in df.columns and 'gender' in df.columns:
            df['sex'] = df['gender']
        return df

    # Merge the RNA table onto the split, build the has_rna mask; fill missing with 0
    def _merge_rna(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, rna_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        rna_data = rna_data.copy()
        rna_data['submitter_id'] = rna_data['submitter_id'].astype(str)
        train_data = pd.merge(train_data, rna_data, on='submitter_id', how='left')
        test_data = pd.merge(test_data, rna_data, on='submitter_id', how='left')

        rna_cols = [col for col in rna_data.columns if col not in ['submitter_id', 'sample']]
        self._rna_cols = rna_cols

        train_data['has_rna'] = (~train_data[rna_cols].isna().all(axis=1)).astype(int)
        test_data['has_rna'] = (~test_data[rna_cols].isna().all(axis=1)).astype(int)

        for col in rna_cols:
            train_data[col] = train_data[col].fillna(0)
            test_data[col] = test_data[col].fillna(0)
        return train_data, test_data, rna_cols

    # Build the has_wsi mask based on whether an embedding exists for the submitter_id
    def _add_wsi_availability(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Add WSI availability flag (patient-level)."""
        if self.wsi_embeddings is None:
            raise ValueError("WSI embeddings not loaded")
        train_data = train_data.copy()
        test_data = test_data.copy()

        # Patient-level: match directly by submitter_id
        train_data['has_wsi'] = train_data['submitter_id'].isin(self.wsi_embeddings.keys()).astype(int)
        test_data['has_wsi'] = test_data['submitter_id'].isin(self.wsi_embeddings.keys()).astype(int)

        return train_data, test_data


    # ======================= Missing modality helpers =======================
    @staticmethod
    def _parse_missing_config(missing_config_train: Any) -> Dict[str, float]:
        """Parse a DisPro-style missing_config string/dict.

        Accepted inputs:
            - "WSI:0.2_RNA:0.4"  (or Omics:0.4)
            - {"WSI":0.2, "RNA":0.4}

        Returns: a unified dict with upper-cased keys, e.g. {"WSI":0.2, "RNA":0.4}
        """
        if missing_config_train is None:
            return {}

        if isinstance(missing_config_train, dict):
            cfg = {str(k).strip().upper(): float(v) for k, v in missing_config_train.items()}
            return cfg

        if isinstance(missing_config_train, str):
            cfg = {}
            parts = [p for p in missing_config_train.split('_') if p]
            for p in parts:
                if ':' in p:
                    k, v = p.split(':', 1)
                elif '=' in p:
                    k, v = p.split('=', 1)
                else:
                    continue
                cfg[k.strip().upper()] = float(v)
            return cfg

        raise TypeError(f"Unsupported missing_config_train type: {type(missing_config_train)}")

    def apply_missing_config(
        self,
        df: pd.DataFrame,
        missing_config_train: Any,
        modalities: Optional[List[str]] = None,
        seed: int = 0,
        complete_cases_only: bool = True,
        disjoint: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        df = df.copy()
        cfg = self._parse_missing_config(missing_config_train)
        if not cfg:
            return df

        if modalities is not None:
            enabled = {m.upper() for m in modalities}
        else:
            enabled = {'WSI', 'RNA'}

        key_to_col = {
            'WSI': 'has_wsi',
            'RNA': 'has_rna',
            'OMICS': 'has_rna',
        }

        active_items = []
        for k, rate in cfg.items():
            if k not in key_to_col:
                continue
            k_std = 'RNA' if k in {'RNA', 'OMICS'} else k
            if k_std not in enabled:
                continue
            active_items.append((k_std, float(rate)))

        if not active_items:
            return df

        rng = np.random.RandomState(seed)

        if complete_cases_only:
            cand_mask = np.ones(len(df), dtype=bool)
            for k_std, _ in active_items:
                col = key_to_col[k_std]
                if col in df.columns:
                    cand_mask &= (df[col].values.astype(int) == 1)
            candidates = df.index[cand_mask].to_numpy()
        else:
            candidates = df.index.to_numpy()

        n_cand = len(candidates)
        if n_cand == 0:
            if verbose:
                print('[apply_missing_config] WARNING: no complete cases to apply missing_config_train.')
            return df

        drop_plan = []
        for k_std, rate in active_items:
            n_drop = int(round(rate * n_cand))
            n_drop = max(0, min(n_drop, n_cand))
            drop_plan.append((k_std, n_drop))

        if disjoint:
            perm = rng.permutation(candidates)
            start = 0
            for k_std, n_drop in drop_plan:
                if n_drop <= 0:
                    continue
                idx_drop = perm[start:start + n_drop]
                start += n_drop
                col = key_to_col[k_std]
                df.loc[idx_drop, col] = 0
        else:
            for k_std, n_drop in drop_plan:
                if n_drop <= 0:
                    continue
                idx_drop = rng.choice(candidates, size=n_drop, replace=False)
                col = key_to_col[k_std]
                df.loc[idx_drop, col] = 0

        if verbose:
            print('[apply_missing_config] Applied missing_config_train:', cfg, 'seed=', seed)
            for k_std, _ in active_items:
                col = key_to_col[k_std]
                if col in df.columns:
                    valid_ratio = df.loc[candidates, col].mean()
                    print(f'  - {k_std}: valid={valid_ratio:.3f}, missing={1-valid_ratio:.3f} (on complete-pair candidates)')

        return df

    # Unified data-prep entry: train.csv/test.csv + rna_data -> {'train','test',...}
    def prepare_data(
        self,
        train_split: pd.DataFrame,
        test_split: pd.DataFrame,
        modalities: List[str],
        rna_data: Optional[pd.DataFrame] = None,
        with_validation: bool = False,
        random_state: int = 1,
        missing_config_train: Optional[Any] = None,
        missing_seed: Optional[int] = None,
        missing_complete_cases_only: bool = True,
        missing_disjoint: bool = True,
        missing_verbose: bool = True,
    ) -> Dict[str, Any]:
        train_data = self._standardize_id(train_split)
        test_data = self._standardize_id(test_split)
        train_data = self._standardize_sex_column(train_data)
        test_data = self._standardize_sex_column(test_data)

        result = {
            'rna_cols': None,
        }

        if 'RNA' in modalities:
            if rna_data is None:
                raise ValueError("rna_data required for RNA modality")
            train_data, test_data, rna_cols = self._merge_rna(train_data, test_data, rna_data)
            result['rna_cols'] = rna_cols

        if 'WSI' in modalities:
            # Patient-level WSI: match directly by submitter_id
            print("Using patient-level WSI embeddings...")
            train_data, test_data = self._add_wsi_availability(train_data, test_data)
            # Deduplicate: keep one row per patient
            train_data = train_data.drop_duplicates(subset='submitter_id', keep='first').copy()
            test_data = test_data.drop_duplicates(subset='submitter_id', keep='first').copy()
            print(f"  Train: {len(train_data)} patients, Test: {len(test_data)} patients")

        # NOTE: in-pipeline validation split is not implemented; callers always pass
        # with_validation=False. The param is kept for API compatibility.
        val_data = None

        # Optional artificial missing modality (DisPro-style)
        if missing_config_train is not None:
            seed = random_state if missing_seed is None else missing_seed
            train_data = self.apply_missing_config(
                train_data,
                missing_config_train=missing_config_train,
                modalities=modalities,
                seed=seed,
                complete_cases_only=missing_complete_cases_only,
                disjoint=missing_disjoint,
                verbose=missing_verbose,
            )

        result['train'] = train_data
        result['val'] = val_data
        result['test'] = test_data
        return result

    # DataFrame -> features/masks/dur/evt (extract per modalities)
    def extract_features(
        self,
        data: pd.DataFrame,
        modalities: List[str],
        rna_cols: Optional[List[str]] = None,
    ) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
        features, masks = {}, {}

        for mod in modalities:
            if mod == 'RNA':
                if rna_cols is None:
                    raise ValueError("rna_cols required for RNA")
                features['RNA'] = data[rna_cols].values.astype('float64')
                masks['RNA'] = data['has_rna'].values.astype('float64')

            elif mod == 'WSI':
                # Patient-level WSI: match directly by submitter_id
                X_wsi = [
                    self.wsi_embeddings.get(pid, np.zeros(self.wsi_dim, dtype='float64'))
                    for pid in data['submitter_id'].values
                ]
                features['WSI'] = np.array(X_wsi)
                masks['WSI'] = data['has_wsi'].values.astype('float64')

        dur = data['dss_survival_days'].values.astype('float64')
        dur = np.where(dur == 0, 1e-16, dur)
        evt = 1 - data['dss_censorship'].values.astype('float64')
        return features, masks, dur, evt

    # Train set: fit scaler + (sort by dur) + PT(log(dur)); returns sort_idx
    def fit_scalers(
        self,
        features: Dict[str, np.ndarray],
        dur: np.ndarray,
        masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict, np.ndarray, np.ndarray]:
        # NOTE: DataPreprocessor may be reused across CV folds.
        # Always refit scalers on the current training split to avoid leakage.
        self.scalers = {}

        transformed = {}
        for mod, X in features.items():
            self.scalers[mod] = StandardScaler()

            has_missing = False
            missing_idx = None
            if masks is not None and mod in masks:
                missing_idx = np.where(masks[mod] == 0)[0]
                if len(missing_idx) > 0:
                    has_missing = True
                    valid_idx = np.where(masks[mod] == 1)[0]

            if has_missing:
                self.scalers[mod].fit(np.asfortranarray(X[valid_idx]))
                X_transformed = self.scalers[mod].transform(X)
                X_transformed[missing_idx, :] = 0.0
                transformed[mod] = X_transformed
            else:
                transformed[mod] = self.scalers[mod].fit_transform(X)

        sort_idx = np.argsort(dur)
        for mod in transformed:
            transformed[mod] = transformed[mod][sort_idx]
        dur_transformed = self.pt.fit_transform(np.log(dur[sort_idx].reshape(-1, 1))).squeeze()
        return transformed, dur_transformed, sort_idx

    # Test set: transform with already-fitted scaler/pt; optionally sort by dur; returns sort_idx
    def transform_data(
        self,
        features: Dict[str, np.ndarray],
        dur: np.ndarray,
        sort: bool = True,
        masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict, np.ndarray, Optional[np.ndarray]]:
        """Transform data with the scaler fitted on the training set (val/test)."""
        transformed = {}
        for mod, X in features.items():
            if mod not in self.scalers:
                raise ValueError(f"Scaler for '{mod}' not fitted")

            has_missing = False
            missing_idx = None
            if masks is not None and mod in masks:
                missing_idx = np.where(masks[mod] == 0)[0]
                if len(missing_idx) > 0:
                    has_missing = True

            X_transformed = self.scalers[mod].transform(X)
            if has_missing:
                X_transformed[missing_idx, :] = 0.0
            transformed[mod] = X_transformed

        if sort:
            sort_idx = np.argsort(dur)
            for mod in transformed:
                transformed[mod] = transformed[mod][sort_idx]
            dur_transformed = self.pt.transform(np.log(dur[sort_idx].reshape(-1, 1))).squeeze()
            return transformed, dur_transformed, sort_idx
        return transformed, dur, None


# features/dur/evt/masks -> DataLoader; iter(loader) yields (inputs_dict, durations, events[, masks_dict])
def create_dataloader(
    features: Dict[str, np.ndarray],
    dur: np.ndarray,
    evt: np.ndarray,
    masks: Optional[Dict[str, np.ndarray]],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from preprocessed data."""
    modality_names = list(features.keys())
    dataset = MultiModalDataset(features, dur, evt, masks)
    collate_fn = get_collate_fn(modality_names, with_mask=masks is not None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)
