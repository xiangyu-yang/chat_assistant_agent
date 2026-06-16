import json
import logging
from pathlib import Path
from typing import List, Any

logger = logging.getLogger(__name__)


def load_json_file(file_path: Path, default: Any = None) -> Any:
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
    return default if default is not None else []


def save_json_file(file_path: Path, data: Any) -> bool:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False


def load_uploaded_files(file_path: Path) -> List[str]:
    return load_json_file(file_path, default=[])


def save_uploaded_files(file_path: Path, files_list: List[str]) -> bool:
    return save_json_file(file_path, files_list)