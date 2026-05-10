import os
from pathlib import Path

# Identify the absolute path to the root of the repository
# (Assuming config.py is located at the root of the repository)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Define standard directory paths based on the new structure
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"

# Ensure experiment and models directories exist so scripts don't fail
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Helper function to easily get the path to a specific dataset or file
def get_data_path(filename: str) -> Path:
    """Returns the absolute path to a file inside the data/ directory."""
    return DATA_DIR / filename

def get_experiment_path(filename: str) -> Path:
    """Returns the absolute path to a file inside the experiments/ directory."""
    return EXPERIMENTS_DIR / filename

def get_model_path(filename: str) -> Path:
    """Returns the absolute path to a file inside the models/ directory."""
    return MODELS_DIR / filename
