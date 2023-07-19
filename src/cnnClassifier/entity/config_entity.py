from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataModelConfig:
    root_dir: Path
    test_dir: Path
    train_dir: Path
    val_dir: Path