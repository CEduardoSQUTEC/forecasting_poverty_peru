from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / 'data'
PROCESSED_PATH = DATA_PATH / 'processed'
RAW_PATH = DATA_PATH / 'raw'

MODELS_PATH = PROJECT_ROOT / 'models'