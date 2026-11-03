"""Rutas del proyecto y carga de parámetros."""
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
VIDEO_DIR = ROOT_DIR / "data" / "video"
FRAMES_DIR = ROOT_DIR / "data" / "frames"
TABLES_DIR = ROOT_DIR / "results" / "tables"
FIGURES_DIR = ROOT_DIR / "results" / "figures"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}


def load_params():
    """Lee config/params.yaml."""
    with open(CONFIG_DIR / "params.yaml") as f:
        return yaml.safe_load(f)


def list_videos():
    """Devuelve los videos de data/video ordenados por nombre."""
    return sorted(p for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
