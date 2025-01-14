from pathlib import Path

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def get_root() -> Path:
    return ROOT_PATH
