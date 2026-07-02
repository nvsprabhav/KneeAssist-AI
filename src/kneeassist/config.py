from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

SAMPLE_DATA_DIR = RAW_DATA_DIR / "sample"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"