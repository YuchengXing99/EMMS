"""
Configuration for EMMS survival prediction.
Centralizes all hyperparameters.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class ModelConfig:
    """Model architecture hyperparameters"""
    K: Dict[str, int] = field(default_factory=lambda: {'RNA': 50, 'WSI': 50})


@dataclass
class LossConfig:
    """Loss-function hyperparameters"""
    c: float = 0.67
    train_lambd: float = 0.75
    nu: float = 1e-16
    xi: float = 0.0005
    rho: float = 0.0005
    sigma: float = 0.1
    fusion_weight: float = 0.01


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 256
    max_epochs: int = 70
    learning_rate: float = 0.011
    milestones: List[int] = field(default_factory=lambda: [50])
    gamma: float = 0.15


@dataclass
class EvalConfig:
    """Evaluation hyperparameters"""
    eval_lambdas: List[float] = field(default_factory=lambda: [
        0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ])


@dataclass
class DataConfig:
    """Data configuration"""
    seed: int = 123
    mmp_root: str = ''
    titan_embeddings_dir: str = ''

    def __post_init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.mmp_root = os.path.join(project_root, 'data')
        self.titan_embeddings_dir = os.path.join(project_root, 'data', 'titan_embeddings')


@dataclass
class MissingConfig:
    """DisPro-style artificial missing-modality configuration"""
    enabled: bool = False
    missing_config_train: Optional[str] = None
    missing_seed: Optional[int] = None
    complete_cases_only: bool = True
    disjoint: bool = True
    verbose: bool = True

    @staticmethod
    def get_dispro_configs() -> List[str]:
        return [
            "WSI:0.0_RNA:0.6",
            "WSI:0.2_RNA:0.4",
            "WSI:0.3_RNA:0.3",
            "WSI:0.4_RNA:0.2",
            "WSI:0.6_RNA:0.0",
        ]


@dataclass
class Config:
    """Top-level config combining all sub-configs"""
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    missing: MissingConfig = field(default_factory=MissingConfig)
    modalities: List[str] = field(default_factory=lambda: ['RNA', 'WSI'])
    use_mask: bool = True
    use_validation: bool = False


def get_wsi_rna_config() -> Config:
    """WSI + RNA dual-modal config - NO VALIDATION"""
    config = Config()
    config.modalities = ['RNA', 'WSI']
    config.use_validation = False
    return config


def get_wsi_rna_with_missing_config(missing_config_train: str = "WSI:0.3_RNA:0.3") -> Config:
    """WSI + RNA dual-modal config + artificial missing modality"""
    config = Config()
    config.modalities = ['RNA', 'WSI']
    config.use_validation = False
    config.missing.enabled = True
    config.missing.missing_config_train = missing_config_train
    config.missing.complete_cases_only = True
    config.missing.disjoint = True
    config.missing.verbose = True
    return config


def get_rna_only_config() -> Config:
    """RNA single-modal config - NO VALIDATION"""
    config = Config()
    config.modalities = ['RNA']
    config.model.K = {'RNA': 50}
    config.use_validation = False
    return config


def get_wsi_only_config() -> Config:
    """WSI single-modal config - NO VALIDATION"""
    config = Config()
    config.modalities = ['WSI']
    config.model.K = {'WSI': 50}
    config.use_validation = False
    return config


def print_config(config: Config):
    """Print configuration summary"""
    print("=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"\n[Modalities]: {config.modalities}")
    print(f"[Use Validation]: {config.use_validation}")
    print(f"[Use Mask]: {config.use_mask}")
    print(f"\n[Model]")
    print(f"  K: {config.model.K}")
    print(f"\n[Loss]")
    print(f"  c: {config.loss.c}")
    print(f"  train_lambd: {config.loss.train_lambd}")
    print(f"  xi: {config.loss.xi}, rho: {config.loss.rho}")
    print(f"  sigma: {config.loss.sigma}")
    print(f"\n[Training]")
    print(f"  learning_rate: {config.training.learning_rate}")
    print(f"  batch_size: {config.training.batch_size}")
    print(f"  max_epochs: {config.training.max_epochs}")
    print(f"  milestones: {config.training.milestones}")
    print(f"  gamma: {config.training.gamma}")
    print(f"\n[Data]")
    print(f"  seed: {config.data.seed}")
    print(f"\n[Missing Modality (DisPro-style)]")
    print(f"  enabled: {config.missing.enabled}")
    if config.missing.enabled:
        print(f"  missing_config_train: {config.missing.missing_config_train}")
        print(f"  complete_cases_only: {config.missing.complete_cases_only}")
        print(f"  disjoint: {config.missing.disjoint}")
    print("=" * 60)


if __name__ == '__main__':
    config = get_wsi_rna_config()
    print_config(config)
