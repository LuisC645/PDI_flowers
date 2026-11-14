"""A4. Seguimiento y estabilización de la geometría.  [BORRADOR: decisiones 4 y 5 sin cerrar]

Como el video ya está grabado, se usan cuadros pasados y futuros:
  1. Atípicos (filtro de Hampel): residuo frente a la mediana móvil > k·MAD local -> el
     cuadro se invalida. (Con MAD global se marcaban giros rápidos reales.)
  2. Huecos cortos (<= max_gap cuadros) se interpolan; los largos quedan inválidos.
  3. Suavizado Savitzky-Golay por tramos continuos.
Nota: el suavizado usa el índice del cuadro, no time_s (el muestreo no es uniforme: ver
frame_index). Revisar si eso importa.
"""
import numpy as np
from scipy.signal import savgol_filter

COLUMNS = ["angle_deg", "center_y", "width_px"]


def flag_outliers(series, window, k):
    """Filtro de Hampel: True donde el valor se aleja de la mediana móvil más de k·MAD local.

    Con el MAD local, un giro rápido pero real (la escala local crece) no se marca como
    atípico; el piso (mitad del MAD típico) evita marcar todo donde la serie es casi constante.
    """
    med = series.rolling(window, center=True, min_periods=3).median()
    resid = (series - med).abs()
    mad = resid.rolling(window, center=True, min_periods=3).median() * 1.4826
    floor = np.nanmedian(mad) * 0.5
    return (resid > k * np.maximum(mad, floor)).fillna(False)


def smooth_segments(series, window, order):
    """Savitzky-Golay sobre cada tramo sin NaN (los tramos cortos solo se promedian)."""
    out = series.copy()
    valid = series.notna()
    seg_id = (valid != valid.shift()).cumsum()
    for _, idx in series[valid].groupby(seg_id[valid]).groups.items():
        seg = series.loc[idx]
        if len(seg) >= window:
            out.loc[idx] = savgol_filter(seg.values, window, order)
        else:
            out.loc[idx] = seg.rolling(5, center=True, min_periods=1).mean()
    return out


def stabilize(geom, cfg):
    """Devuelve la geometría estabilizada y el estado de cada cuadro.

    status: measured (medido y válido), interpolated (hueco corto) o invalid.
    """
    out = geom[["frame"]].copy()
    ok = geom["valid"].astype(bool).copy()
    for col in COLUMNS:  # primero se quitan los atípicos de cualquier variable
        s = geom[col].where(ok)
        ok &= ~flag_outliers(s, cfg["median_window"], cfg["mad_k"])
    for col in COLUMNS:
        s = geom[col].where(ok).interpolate(limit=cfg["max_gap"], limit_area="inside")
        out[col] = smooth_segments(s, cfg["savgol_window"], cfg["savgol_order"])
    out["center_x"] = geom["center_x"]
    out["status"] = np.where(ok, "measured", np.where(out["width_px"].notna(), "interpolated",
                                                      "invalid"))
    return out
