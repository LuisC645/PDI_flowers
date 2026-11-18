"""Rutas del proyecto y carga de parámetros."""
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
RAW_VIDEO_DIR = DATA_DIR / "videos_raw"              # videos originales (entrada)
PROCESSED_VIDEO_DIR = DATA_DIR / "videos_processed"  # videos estandarizados (trabajo)
VIDEOS_METADATA = DATA_DIR / "videos.csv"            # fecha, variedad y cama de cada video
NORMALIZED_DIR = DATA_DIR / "normalized"
MASKS_DIR = NORMALIZED_DIR / "masks"                 # máscaras de la cama (A2)
TRANSFORMS_DIR = NORMALIZED_DIR / "transforms"       # geometría y transformaciones (A3-A6)
NORMALIZED_VIDEO_DIR = NORMALIZED_DIR / "video"      # video normalizado (A5-A6)
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
TABLES_DIR = ROOT_DIR / "results" / "tables"
FIGURES_DIR = ROOT_DIR / "results" / "figures"
METRICS_DIR = ROOT_DIR / "results" / "metrics"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}


def load_params():
    """Lee config/params.yaml."""
    with open(CONFIG_DIR / "params.yaml") as f:
        return yaml.safe_load(f)


def excluded_videos():
    """Videos marcados en la columna `exclude` de data/videos.csv (por ejemplo, duplicados)."""
    import pandas as pd
    if not VIDEOS_METADATA.exists():
        return set()
    meta = pd.read_csv(VIDEOS_METADATA, dtype=str).fillna("")
    if "exclude" not in meta:
        return set()
    return set(meta.video_id[meta.exclude.str.strip() != ""])


def list_videos(folder=PROCESSED_VIDEO_DIR):
    """Videos de una carpeta, ordenados por nombre, sin los excluidos en data/videos.csv."""
    skip = excluded_videos()
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in VIDEO_EXTENSIONS and p.stem not in skip)
