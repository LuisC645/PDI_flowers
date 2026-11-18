"""Compara configuraciones del Bloque A (A2-A4) sobre TODOS los videos a la vez.

Sirve para elegir un solo modo de trabajo que funcione en todos los casos, en lugar de
ajustar video por video. Cada configuración es un conjunto de cambios sobre params.yaml.
Corre A2/A3 y A4 en paralelo (un proceso por video) y escribe en una carpeta temporal:
no toca data/normalized ni results.

Métricas por video (no necesitan referencia a mano):
  valid_pct    cuadros con geometría medida en A3 (incluye partial)
  partial_pct  cuadros medidos con un solo borde
  bad_pct      inválidos + interpolados después de A4 (meta A7: <= 10 %)
  angle_noise  desviación de (medido - estabilizado) del ángulo, en grados
  width_noise  igual para el ancho, en % del ancho medio
Uso:  python scripts/sweep_block_a.py                       (configuraciones de CONFIGS)
      python scripts/sweep_block_a.py --configs base slope  (solo algunas)
"""
import argparse
import contextlib
import copy
import io
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common.paths import TABLES_DIR, list_videos, load_params  # noqa: E402

CONFIGS = {
    "base": {},
    "slope": {"bed.prior_slope_tol_deg": 8},
    "partial": {"bed.allow_partial": True, "bed.prior_slope_tol_deg": 8},
    "partial_wide": {"bed.allow_partial": True, "bed.prior_slope_tol_deg": 8,
                     "bed.max_edge_angle_diff_deg": 15},
    "p_step1": {"bed.allow_partial": True, "bed.prior_slope_tol_deg": 8, "bed.edge_step": 1},
    "p_ema03": {"bed.allow_partial": True, "bed.prior_slope_tol_deg": 8, "bed.score_ema": 0.3},
    "p_tol5": {"bed.allow_partial": True, "bed.prior_slope_tol_deg": 8, "bed.ransac_tol": 5},
    "p_slope5": {"bed.allow_partial": True, "bed.prior_slope_tol_deg": 5},
    "c_slope5_step1": {"bed.allow_partial": True, "bed.prior_slope_tol_deg": 5, "bed.edge_step": 1},
    "c_slope5_ema03": {"bed.allow_partial": True, "bed.prior_slope_tol_deg": 5, "bed.score_ema": 0.3},
    "c_all": {"bed.allow_partial": True, "bed.prior_slope_tol_deg": 5, "bed.score_ema": 0.3,
              "bed.edge_step": 1},
}


def apply(params, changes):
    """Copia de params con los cambios {"seccion.clave": valor}."""
    p = copy.deepcopy(params)
    for key, value in changes.items():
        sec, k = key.split(".")
        p[sec][k] = value
    return p


def run_one(args):
    name, changes, video, out_dir = args
    import block_a  # se importa en el proceso hijo
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    block_a.TRANSFORMS_DIR = block_a.MASKS_DIR = block_a.FIGURES_DIR = out
    P = apply(load_params(), changes)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        block_a.step_a23(video, P)
        block_a.step_a4(video, P)
    raw = pd.read_csv(out / f"{video.stem}_geometry_raw.csv")
    geom = pd.read_csv(out / f"{video.stem}_geometry.csv")
    m = geom.status.isin(["measured", "partial"]) & raw.valid
    full = geom.status == "measured"
    return {
        "config": name, "video": video.stem,
        "valid_pct": round(100 * raw.valid.mean(), 1),
        "partial_pct": round(100 * (geom.status == "partial").mean(), 1),
        "bad_pct": round(100 * geom.status.isin(["invalid", "interpolated"]).mean(), 1),
        "angle_noise": round((raw.angle_deg - geom.angle_deg)[m].std(), 2),
        "width_noise": round(100 * (raw.width_px - geom.width_px)[full].std()
                             / geom.width_px[full].mean(), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    videos = list_videos()
    with tempfile.TemporaryDirectory() as tmp:
        jobs = [(c, CONFIGS[c], v, tmp) for c in args.configs for v in videos]
        with ProcessPoolExecutor(args.workers) as ex:
            rows = list(ex.map(run_one, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "sweep_block_a.csv", index=False)
    cols = ["valid_pct", "partial_pct", "bad_pct", "angle_noise", "width_noise"]
    print(df.pivot(index="video", columns="config", values="bad_pct").to_string())
    summary = df.groupby("config")[cols].mean().round(2)
    summary["worst_bad_pct"] = df.groupby("config").bad_pct.max()
    summary["videos_bad_le_10"] = df.assign(ok=df.bad_pct <= 10).groupby("config").ok.sum()
    print(summary.loc[args.configs].to_string())


if __name__ == "__main__":
    main()
