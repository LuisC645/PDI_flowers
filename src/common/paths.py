"""Rutas del proyecto y carga de parámetros."""
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
RAW_VIDEO_DIR = ROOT_DIR / "data" / "videos_raw"              # videos originales (entrada)
PROCESSED_VIDEO_DIR = ROOT_DIR / "data" / "videos_processed"  # videos estandarizados (trabajo)
VIDEOS_METADATA = ROOT_DIR / "data" / "videos.csv"            # fecha, variedad y cama de cada video
TABLES_DIR = ROOT_DIR / "results" / "tables"
FIGURES_DIR = ROOT_DIR / "results" / "figures"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}


def load_params():
    """Lee config/params.yaml."""
    with open(CONFIG_DIR / "params.yaml") as f:
        return yaml.safe_load(f)


def list_videos(folder=PROCESSED_VIDEO_DIR):
    """Devuelve los videos de una carpeta ordenados por nombre."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
