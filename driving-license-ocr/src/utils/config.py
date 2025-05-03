"""
Configuration settings for the License OCR pipeline.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import os
import yaml # type: ignore


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = DATA_DIR / "outputs/tables"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "raw").mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "preprocessing": {
        "resize_width": 1200,
        "min_text_height": 12,
        "threshold_method": "adaptive",  # 'otsu', 'adaptive', 'canny'
        "denoise": True,
        "deskew": True,
        "contrast_enhancement": True
    },
    "ocr": {
        "engine": "tesseract",
        "lang": "en",
        "config": "--psm 7 --oem 3",
        "min_confidence": 60,
    },
    "extraction": {
        "vehicle_classes": ["A1", "A", "B1", "B", "C1", "C", "CE", "D1", "D", "DE", "G1", "G", "J"],
        "date_formats": ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"],
        "default_locale": "en_US"
    },
    "evaluation": {
        "metrics": ["accuracy", "precision", "recall", "f1"],
        "class_weights": {
            "vehicle_class": 0.4,
            "start_date": 0.3,
            "expiry_date": 0.3
        }
    }
}

