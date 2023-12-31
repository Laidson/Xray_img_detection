from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataModelConfig:
    root_dir: Path
    test_dir: Path
    train_dir: Path
    val_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass(frozen=True)
class PrepareCallBacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    validation_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    validation_data: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int

@dataclass(frozen=True)
class PredictConfig:
    path_of_model: Path
    params_image_size: list
    params_batch_size: int

@dataclass(frozen=True)
class ModelProdConfig:
    path_to_model: Path
    prod_model_path: Path