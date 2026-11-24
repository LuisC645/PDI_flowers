"""Rutas del proyecto y carga de parámetros."""
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
RAW_VIDEO_DIR = DATA_DIR / "videos_raw"              # videos originales (entrada)
PROCESSED_VIDEO_DIR = DATA_DIR / "videos_processed"  # videos estandarizados (trabajo)
VIDEOS_METADATA = DATA_DIR / "videos.csv"            # cama, color y fecha de grabación de cada video
COMPANY_DIR = DATA_DIR / "ground_truth" / "company"  # datos de la empresa (no se suben a git)
METADATA_COLUMNS = ["video_id", "bed_id", "flower_color", "record_date", "rotate", "exclude"]
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


def load_metadata(include_excluded=False):
    """Lee y valida data/videos.csv; devuelve un DataFrame indexado por video_id.

    Falla con un mensaje claro si faltan columnas, hay ids repetidos, un color no tiene perfil
    en params.yaml o la fecha de grabación no es AAAA-MM-DD."""
    import pandas as pd
    if not VIDEOS_METADATA.exists():
        raise SystemExit(f"Missing {VIDEOS_METADATA}: one row per video ({', '.join(METADATA_COLUMNS)})")
    meta = pd.read_csv(VIDEOS_METADATA, dtype=str).fillna("")
    meta.columns = meta.columns.str.strip()
    for c in ["rotate", "exclude", "record_date"]:   # opcionales
        if c not in meta:
            meta[c] = ""
    missing = {"video_id", "bed_id", "flower_color"} - set(meta.columns)
    if missing:
        raise SystemExit(f"{VIDEOS_METADATA.name}: missing columns {sorted(missing)}")
    meta = meta.apply(lambda s: s.str.strip())
    dup = meta.video_id[meta.video_id.duplicated()].tolist()
    if dup:
        raise SystemExit(f"{VIDEOS_METADATA.name}: repeated video_id {dup}")
    if not include_excluded:
        meta = meta[meta.exclude == ""]
    profiles = set(load_params()["profiles"])
    bad = meta[(meta.flower_color != "") & ~meta.flower_color.isin(profiles)]
    if len(bad):
        raise SystemExit(f"{VIDEOS_METADATA.name}: flower_color without a profile in params.yaml: "
                         f"{dict(zip(bad.video_id, bad.flower_color))} (profiles: {sorted(profiles)})")
    dates = pd.to_datetime(meta.record_date.replace("", None), format="%Y-%m-%d", errors="coerce")
    bad = meta.video_id[(meta.record_date != "") & dates.isna()].tolist()
    if bad:
        raise SystemExit(f"{VIDEOS_METADATA.name}: record_date must be YYYY-MM-DD in {bad}")
    return meta.set_index("video_id", drop=False)


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
