"""Calibración del color de flor madura, por cama (decisión 8).

Sin ejemplos etiquetados no se puede saber qué color corresponde a "lista para cortar". Se
probó separar en dos grupos con Otsu juntando las camas de cada color, y no funcionó: en
rosada, roja y blanca la distribución tiene un solo pico (el corte no separa nada real), y en
naranja los dos picos son dos variedades (63-40A más saturada que 63_39A y 63_41A), no dos
grados de madurez.

Criterio adoptado (conservador): el color se calibra POR CAMA. Una flor tiene color "no
maduro" solo si es claramente más pálida que el resto de su propia cama:
    índice < mediana - k·MAD   (k = color_mad_k, MAD escalado 1.4826)
índice = saturación (flores de color) o brillo - saturación (flor blanca). Así se descuentan
la variedad y la luz de cada grabación, y el color solo descarta casos claros. Cuando existan
flores etiquetadas a mano, el umbral se elige con curva ROC.

Entrada: results/tables/flowers_<v>.csv (rastros de >= 10 cuadros, después de block_b b3)
Salida:  results/metrics/color_calibration.json, results/figures/b4_color_calibration.png
Uso:     python scripts/calibrate_colors.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.counting import color_index  # noqa: E402
from src.common.paths import (FIGURES_DIR, METRICS_DIR, TABLES_DIR, VIDEOS_METADATA,  # noqa: E402
                              load_params)


def main():
    k = load_params()["cut"]["color_mad_k"]
    meta = pd.read_csv(VIDEOS_METADATA, dtype=str).fillna("")
    meta = meta[(meta.exclude.str.strip() == "") & (meta.flower_color != "")]
    out, hists = {}, []
    for v, color in zip(meta.video_id, meta.flower_color):
        path = TABLES_DIR / f"flowers_{v}.csv"
        if not path.exists():
            continue
        idx = color_index(pd.read_csv(path), color)
        med = float(idx.median())
        mad = float(1.4826 * np.median(np.abs(idx - med)))
        thr = med - k * mad
        out[v] = {"color": color, "feature": "value - saturation" if color == "white" else "saturation",
                  "median": round(med, 1), "mad": round(mad, 1), "threshold": round(thr, 1),
                  "not_mature_pct": round(float(100 * (idx < thr).mean()), 1), "n_flowers": int(len(idx))}
        hists.append((v, color, idx, thr))
        print(v, out[v])
    (METRICS_DIR / "color_calibration.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    fig, axes = plt.subplots(2, (len(hists) + 1) // 2, figsize=(3.6 * ((len(hists) + 1) // 2), 5.5))
    for ax, (v, color, idx, thr) in zip(axes.flat, hists):
        ax.hist(idx, bins=50, color="0.6")
        ax.axvline(thr, color="k", ls="--")
        ax.set_title(f"{v} ({color}): {thr:.0f}", fontsize=9)
        ax.set_xlabel(out[v]["feature"], fontsize=8)
    for ax in list(axes.flat)[len(hists):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "b4_color_calibration.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
