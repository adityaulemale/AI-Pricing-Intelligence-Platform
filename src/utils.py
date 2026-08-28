import os
import pickle
from pathlib import Path


def save_object(file_path: str, obj):
    """
    Save a Python object to disk using pickle.
    """

    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def load_object(file_path: str):
    """
    Load a Python object from a pickle file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)


def create_directories(directory_paths: list):
    """
    Create multiple directories if they do not exist.
    """

    for directory_path in directory_paths:
        os.makedirs(
            directory_path,
            exist_ok=True
        )

