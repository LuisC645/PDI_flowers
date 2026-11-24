"""Comparación con los datos de la empresa (validación indirecta).

La empresa reporta **tallos recibidos en poscosecha por cama y por día**, no flores: un tallo de
micropompón lleva varias flores. Por eso no se mide el error absoluto del conteo de flores, sino:

1. Tallos estimados = flores contadas / (flores por tallo). El factor flores/tallo sale de las
   camas válidas, en jerarquía: misma variedad → mismo tipo de flor del reporte de la empresa
   (columna flower_type) → todas. Su error se mide dejando una cama fuera (el factor se calcula
   sin la cama que se predice).
2. Momento del corte: fracción acumulada de flores listas por día según el pipeline frente a la
   fracción acumulada de tallos recibidos, en días desde la grabación (`record_date`). Se calibra
   el diámetro de corte que reproduce el día mediano de corte de la empresa.

Una cama queda fuera de la calibración si no está en el reporte, si el reporte empieza después
de su primer día de corte (`cut_start_date`) o si tiene más de una variedad en el reporte.

Entrada (datos de la empresa, ignorados por git, en data/ground_truth/company/):
    reception_report.csv   el Excel "Reporte mov. recepcion" convertido a CSV
    beds_company.csv       video_id, variety, cut_start_date, cut_end_date, note
y del pipeline: data/videos.csv, results/metrics/summary_<video>.json, flowers_line_<video>.csv
Salida (ignorada por git): results/tables/company_*.csv, results/tables/beds_flowers_stems.csv,
    results/metrics/company_comparison.json; figuras results/figures/company_*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.counting import classify_maturity  # noqa: E402
from src.common.paths import (COMPANY_DIR, FIGURES_DIR, METRICS_DIR, TABLES_DIR,  # noqa: E402
                              load_metadata, load_params)

REPORT = COMPANY_DIR / "reception_report.csv"
BEDS_COMPANY = COMPANY_DIR / "beds_company.csv"
MAX_DAY = 8          # horizonte de la comparación de tiempos (días desde la grabación)
LEVELS = ("variety", "flower_type")   # jerarquía del factor flores/tallo (después: todas)
INK, MUTED, GRID = "#2b2b29", "#76756f", "#e4e3de"
C_PRED, C_COMP = "#2a78d6", "#eb6834"   # pipeline, empresa
FLAGS_ES = {"not in report": "no está en el reporte",
            "report starts after first cut": "faltan los primeros cortes",
            "mixed varieties": "variedades mezcladas"}


# --------------------------------------------------------------------------- entrada

def load():
    for path, how in [(REPORT, "convert the company Excel to CSV (see README)"),
                      (BEDS_COMPANY, "video_id, variety, cut_start_date, cut_end_date, note")]:
        if not path.exists():
            raise SystemExit(f"Missing {path.relative_to(COMPANY_DIR.parents[2])}: {how}")
    meta = load_metadata().reset_index(drop=True)
    comp = pd.read_csv(BEDS_COMPANY, dtype=str).fillna("")
    meta = meta.merge(comp, on="video_id", how="left").fillna("")
    rep = pd.read_csv(REPORT, parse_dates=["reception_date"])
    need = {"bed_id", "flower_type", "variety", "reception_date", "stems"}
    if need - set(rep.columns):
        raise SystemExit(f"{REPORT.name}: missing columns {sorted(need - set(rep.columns))}")
    summ, missing = {}, []
    for v in meta.video_id:
        p = METRICS_DIR / f"summary_{v}.json"
        if p.exists():
            summ[v] = json.loads(p.read_text())
        else:
            missing.append(v)
    if missing:
        print(f"WARNING: no summary (run block_b.py) for {missing}; they are left out")
    no_date = meta.video_id[(meta.record_date == "") & meta.video_id.isin(summ)].tolist()
    if no_date:
        raise SystemExit(f"videos.csv: record_date is needed for {no_date}")
    return meta[meta.video_id.isin(summ)], rep, summ


def weighted_median(days, w):
    """Mediana de `days` ponderada por `w` (el día en que se recibió la mitad de los tallos)."""
    days, w = np.asarray(days), np.asarray(w, dtype=float)
    order = np.argsort(days)
    d, c = days[order], np.cumsum(w[order])
    return int(d[np.searchsorted(c, c[-1] / 2)])


def predicted_cumulative(s):
    """Fracción acumulada de flores listas al día d (0 = día de la grabación)."""
    by_day = np.zeros(MAX_DAY + 1)
    by_day[0], by_day[1] = s["today"], s["tomorrow"]
    for k, n in s.get("later_by_days", {}).items():
        by_day[min(int(k), MAX_DAY)] += n   # "1" en later_by_days: pasan a mañana por color
    return np.cumsum(by_day) / s["total_flowers"]


def bed_rows(meta, rep, summ):
    first_report_day = rep.reception_date.min()
    rows, timing = [], []
    for m in meta.itertuples():
        s = summ[m.video_id]
        r = rep[rep.bed_id == m.bed_id]
        rec = pd.Timestamp(m.record_date)
        varieties = sorted(r.variety.unique())
        flags = []
        if r.empty:
            flags.append("not in report")
        if m.cut_start_date and pd.Timestamp(m.cut_start_date) < first_report_day:
            flags.append("report starts after first cut")
        if len(varieties) > 1:
            flags.append("mixed varieties")
        stems = int(r.stems.sum())
        days = (r.reception_date - rec).dt.days
        rows.append({
            "video": m.video_id, "bed_id": m.bed_id, "variety": m.variety, "color": m.flower_color,
            "flower_type": r.flower_type.mode().iat[0] if len(r) else "",
            "report_varieties": " + ".join(varieties), "flowers": s["total_flowers"],
            "flowers_other_method": s["total_flowers_other_method"], "stems": stems,
            "flowers_per_stem": round(s["total_flowers"] / stems, 2) if stems else np.nan,
            "flowers_per_stem_other": round(s["total_flowers_other_method"] / stems, 2) if stems else np.nan,
            "pred_today_pct": s["today_pct"], "pred_tomorrow_pct": s["tomorrow_pct"],
            "pred_later_pct": s["later_pct"],
            "company_day0_pct": round(100 * r.stems[days <= 0].sum() / stems, 1) if stems else np.nan,
            "company_day1_pct": round(100 * r.stems[days == 1].sum() / stems, 1) if stems else np.nan,
            "company_first_day": int(days.min()) if stems else np.nan,
            "company_median_day": weighted_median(days, r.stems) if stems else np.nan,
            "pred_median_day": int(np.argmax(predicted_cumulative(s) >= 0.5)),
            "valid": not flags, "flags": "; ".join(flags)})
        if stems:
            comp = np.array([r.stems[days <= d].sum() / stems for d in range(MAX_DAY + 1)])
            pred = predicted_cumulative(s)
            for d in range(MAX_DAY + 1):
                timing.append({"video": m.video_id, "bed_id": m.bed_id, "day": d,
                               "date": (rec + pd.Timedelta(days=d)).date(),
                               "pred_cum_pct": round(100 * pred[d], 1),
                               "company_cum_pct": round(100 * comp[d], 1), "valid": not flags})
    df = pd.DataFrame(rows)
    if not df.valid.any():
        raise SystemExit("No valid bed to calibrate flowers per stem (see the flags in the table)")
    return df, pd.DataFrame(timing)


# --------------------------------------------------------------------------- tallos

def flowers_per_stem_factor(df, col, row, skip=None, levels=LEVELS):
    """Flores por tallo de las camas válidas (sin `skip`), en jerarquía: el primer nivel de
    `levels` con al menos una cama igual a `row`; si ninguno, todas las camas válidas."""
    v = df[df.valid & (df.video != skip)]
    for level in levels:
        same = v[v[level] == row[level]]
        if row[level] and len(same):
            return float((same[col] / same.stems).mean()), level, len(same)
    return float((v[col] / v.stems).mean()), "all", len(v)


def stem_errors(pred, true):
    """Métricas de los tallos estimados frente a los de la empresa."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    e = 100 * (pred - true) / true
    return {"n_beds": int(len(e)), "mape_pct": round(float(np.abs(e).mean()), 1),
            "median_ape_pct": round(float(np.median(np.abs(e))), 1),
            "bias_pct": round(float(e.mean()), 1), "rmse_pct": round(float(np.sqrt((e ** 2).mean())), 1),
            "max_abs_pct": round(float(np.abs(e).max()), 1),
            "within_10_pct": int((np.abs(e) <= 10).sum()), "within_20_pct": int((np.abs(e) <= 20).sum()),
            "total_error_pct": round(float(100 * (pred.sum() - true.sum()) / true.sum()), 1)}


def leave_one_out(df, col, levels=LEVELS):
    v = df[df.valid]
    pred = [getattr(r, col) / flowers_per_stem_factor(df, col, r._asdict(), skip=r.video, levels=levels)[0]
            for r in v.itertuples()]
    return np.array(pred), v.stems.values


def estimate_stems(df):
    """`est_*`: factor con todas las camas válidas (lo que se entrega).
    `loo_*`: solo en camas válidas, factor sin la propia cama (error honesto de la conversión)."""
    rows = []
    for r in df.itertuples():
        row_d = r._asdict()
        k, src, n = flowers_per_stem_factor(df, "flowers", row_d)
        row = {"video": r.video, "factor": round(k, 2), "factor_source": src, "factor_beds": n,
               "est_stems": round(r.flowers / k)}
        for stage in ["today", "tomorrow", "later"]:
            row[f"est_stems_{stage}"] = round(r.flowers * getattr(r, f"pred_{stage}_pct") / 100 / k)
        if r.valid:
            for col, tag in [("flowers", "line"), ("flowers_other_method", "tracking")]:
                k_loo, src_loo, _ = flowers_per_stem_factor(df, col, row_d, skip=r.video)
                row[f"loo_stems_{tag}"] = round(getattr(r, col) / k_loo)
                row[f"loo_error_pct_{tag}"] = round(100 * (getattr(r, col) / k_loo - r.stems) / r.stems, 1)
                row["loo_factor_source"] = src_loo
        rows.append(row)
    return df.merge(pd.DataFrame(rows), on="video")


def same_variety_only(df):
    """Tallos estimados SOLO con camas válidas de la misma variedad (sin la propia cama).

    Se evalúa en toda cama con tallos de la empresa que tenga al menos otra cama válida de su
    variedad (incluye camas no válidas, que así son una prueba fuera de la calibración). Las
    variedades con una sola cama válida no se pueden evaluar así."""
    rows = []
    for r in df[df.stems > 0].itertuples():
        same = df[df.valid & (df.variety == r.variety) & (df.video != r.video)]
        if not len(same):
            continue
        for col, tag in [("flowers", "line"), ("flowers_other_method", "tracking")]:
            k = float((same[col] / same.stems).mean())
            rows.append({"bed_id": r.bed_id, "variety": r.variety, "method": tag, "factor_beds": len(same),
                         "valid": r.valid, "est_stems": round(getattr(r, col) / k), "stems": r.stems})
    t = pd.DataFrame(rows)
    if t.empty:
        return {"note": "no variety with two valid beds"}
    t["error_pct"] = (100 * (t.est_stems - t.stems) / t.stems).round(1)
    return {"note": "factor only from other valid beds of the same variety",
            "metrics": {m: stem_errors(g.est_stems, g.stems) for m, g in t.groupby("method")},
            "per_bed": t.to_dict(orient="records")}


def factor_sensitivity(df):
    """Error (dejando una cama fuera) con otras formas de agrupar el factor flores/tallo."""
    out = {}
    for name, levels in [("variety > flower_type > all", LEVELS), ("variety > all", ("variety",)),
                         ("flower_type > all", ("flower_type",)), ("all", ())]:
        out[name] = {tag: stem_errors(*leave_one_out(df, col, levels))["mape_pct"]
                     for col, tag in [("flowers", "line"), ("flowers_other_method", "tracking")]}
    return out


# --------------------------------------------------------------------------- cuándo se corta

def flowers_table(video):
    return pd.read_csv(TABLES_DIR / f"flowers_line_{video}.csv")


def median_ready_day(f, d_cut, g):
    """Día mediano en que las flores alcanzan d_cut (solo tamaño, crecimiento g por día)."""
    return float(np.median(np.ceil(np.log(d_cut / f.d_cm) / np.log(1 + g)).clip(lower=0)))


def calibrate_cut(df, P):
    """Diámetro de corte que reproduce el día mediano de corte de la empresa.

    En cada cama válida: d_req = mediana(d · (1+g)^D), con D el día mediano (ponderado por
    tallos) en que la empresa recibió la cama: la mitad de las flores alcanza d_req el día D.
    El valor para todas las camas es la mediana de d_req. Se valida dejando una cama fuera.
    """
    g, coin = P["cut"]["growth_per_day"], P["cut"]["coin_diameter_cm"]
    rows = []
    for r in df[df.valid].itertuples():
        f = flowers_table(r.video)
        rows.append({"video": r.video, "variety": r.variety, "company_median_day": int(r.company_median_day),
                     "d_median_cm": round(float(f.d_cm.median()), 2),
                     "d_req_cm": round(float(np.median(f.d_cm * (1 + g) ** r.company_median_day)), 2)})
    t = pd.DataFrame(rows)
    for i, r in t.iterrows():
        f = flowers_table(r.video)
        d_loo = float(t.d_req_cm.drop(i).median()) if len(t) > 1 else float(r.d_req_cm)
        t.loc[i, "d_cut_loo_cm"] = round(d_loo, 2)
        t.loc[i, "pred_day_coin"] = median_ready_day(f, coin, g)
        t.loc[i, "pred_day_loo"] = median_ready_day(f, d_loo, g)
    t["err_days_coin"] = t.pred_day_coin - t.company_median_day
    t["err_days_loo"] = t.pred_day_loo - t.company_median_day
    d_cal = float(t.d_req_cm.median())
    return t, {"growth_per_day_assumed": g, "coin_cm": coin, "d_cut_calibrated_cm": round(d_cal, 2),
               "ratio_to_coin": round(d_cal / coin, 2),
               "d_req_range_cm": [float(t.d_req_cm.min()), float(t.d_req_cm.max())],
               "mae_days_coin": round(float(t.err_days_coin.abs().mean()), 1),
               "mae_days_calibrated_loo": round(float(t.err_days_loo.abs().mean()), 1),
               "bias_days_coin": round(float(t.err_days_coin.mean()), 1),
               "bias_days_calibrated_loo": round(float(t.err_days_loo.mean()), 1)}


def calibrated_stages(df, P, d_cal):
    """Hoy / mañana / otros días con el diámetro calibrado (mismo color por cama que B4)."""
    cal = json.loads((METRICS_DIR / "color_calibration.json").read_text())
    ratio = d_cal / P["cut"]["coin_diameter_cm"]
    rows, cum = [], {}
    for r in df.itertuples():
        f = classify_maturity(flowers_table(r.video), r.color, cal[r.video]["threshold"], P["cut"], ratio)
        n = f.stage.value_counts()
        row = {"video": r.video}
        for st in ["today", "tomorrow", "later"]:
            row[f"cal_{st}_pct"] = round(100 * n.get(st, 0) / len(f), 1)
            row[f"cal_est_stems_{st}"] = round(r.flowers * n.get(st, 0) / len(f) / r.factor)
        rows.append(row)
        day = np.where(f.stage == "today", 0, np.where(f.stage == "tomorrow", 1, f.days_to_size.clip(lower=2)))
        cum[r.video] = np.array([(day <= d).mean() for d in range(MAX_DAY + 1)])
    return df.merge(pd.DataFrame(rows), on="video"), cum


def group_stats(df, by):
    g = df[df.valid].groupby(by)
    st = g.agg(beds=("video", "count"), flowers_per_stem=("flowers_per_stem", "mean"),
               sd=("flowers_per_stem", "std"),
               flowers_per_stem_other=("flowers_per_stem_other", "mean"),
               sd_other=("flowers_per_stem_other", "std"))
    st["cv_pct"] = 100 * st.sd / st.flowers_per_stem
    st["cv_other_pct"] = 100 * st.sd_other / st.flowers_per_stem_other
    return st.round(2)


# --------------------------------------------------------------------------- figuras

def style(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def short(variety):
    return variety.split("-", 1)[-1] if variety else "?"


def fig_ratio(df):
    d = df[df.stems > 0].sort_values(["valid", "flower_type", "flowers_per_stem"], ascending=[True, False, True])
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(d) + 1.2))
    y = np.arange(len(d))
    ax.barh(y, d.flowers_per_stem, height=0.6, color=[C_PRED if v else "#b7d3f6" for v in d.valid],
            hatch=["" if v else "///" for v in d.valid], edgecolor="white", lw=0)
    for yi, r in zip(y, d.itertuples()):
        note = "" if r.valid else "  (" + "; ".join(FLAGS_ES.get(f, f) for f in r.flags.split("; ")) + ")"
        ax.text(r.flowers_per_stem + 0.1, yi, f"{r.flowers_per_stem:.1f}{note}",
                va="center", fontsize=8.5, color=INK if r.valid else MUTED)
    ax.set_yticks(y, [f"{r.bed_id} · {short(r.variety)} · {r.flower_type.split()[-1]}" for r in d.itertuples()],
                  color=INK)
    ax.set_xlabel("flores contadas por tallo recibido", color=MUTED, fontsize=9)
    ax.set_xlim(0, d.flowers_per_stem.max() * 1.6)
    ax.grid(axis="y", visible=False)
    style(ax)
    ax.set_title("Flores contadas por tallo cosechado (rayado: cama fuera de la calibración)",
                 loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "company_flowers_per_stem.png", dpi=130)
    plt.close(fig)


def fig_stems(df):
    """Tallos estimados (dejando la cama fuera) frente a los de la empresa, en las camas válidas."""
    v = df[df.valid]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lim = 1.15 * max(v.stems.max(), v.loo_stems_line.max())
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", label="igual a la empresa")
    ax.fill_between([0, lim], [0, 0.8 * lim], [0, 1.2 * lim], color=GRID, alpha=0.6, lw=0, label="±20 %")
    ax.scatter(v.stems, v.loo_stems_line, s=60, color=C_PRED, edgecolor="white", lw=2, zorder=3,
               label="pipeline (línea de conteo)")
    for r in v.itertuples():
        ax.annotate(f"{r.bed_id}  {r.loo_error_pct_line:+.0f} %", (r.stems, r.loo_stems_line),
                    xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("tallos de la empresa", color=MUTED, fontsize=9)
    ax.set_ylabel("tallos estimados (factor sin la propia cama)", color=MUTED, fontsize=9)
    ax.set_title("Tallos estimados frente a la empresa", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "company_stems_loo.png", dpi=130)
    plt.close(fig)


def fig_timing(tim, df, cum_cal, d_cal, coin):
    beds = df[df.stems > 0].video.tolist()
    n = len(beds)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(3.2 * cols, 2.5 * np.ceil(n / cols)),
                             sharex=True, sharey=True, squeeze=False)
    for ax, v in zip(axes.flat, beds):
        t = tim[tim.video == v]
        r = df[df.video == v].iloc[0]
        ax.plot(t.day, t.pred_cum_pct, color=C_PRED, lw=2, marker="o", ms=4,
                label=f"pipeline, moneda {coin:.2f} cm: flores listas")
        ax.plot(t.day, 100 * cum_cal[v], color=C_PRED, lw=2, ls="--", alpha=0.75,
                label=f"pipeline, calibrado {d_cal:.2f} cm")
        ax.step(t.day, t.company_cum_pct, where="post", color=C_COMP, lw=2, label="empresa: tallos cortados")
        ax.set_title(f"{r.bed_id} · {short(r.variety)}" + ("" if r.valid else " *"),
                     fontsize=9, color=INK if r.valid else MUTED, loc="left")
        ax.set_ylim(0, 105)
        ax.set_xticks(range(0, MAX_DAY + 1, 2))
        style(ax)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("días desde la grabación", color=MUTED, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("% acumulado", color=MUTED, fontsize=8.5)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.945), frameon=False, fontsize=9, ncol=3)
    fig.suptitle("¿Cuándo se corta? Pipeline frente a recepción de la empresa  (* cama fuera de la calibración)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIGURES_DIR / "company_timing.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main

def main():
    P = load_params()
    meta, rep, summ = load()
    df, tim = bed_rows(meta, rep, summ)
    df = estimate_stems(df)
    cal_t, cal = calibrate_cut(df, P)
    df, cum_cal = calibrated_stages(df, P, cal["d_cut_calibrated_cm"])
    v = df[df.valid]

    df.to_csv(TABLES_DIR / "company_comparison.csv", index=False)
    tim.to_csv(TABLES_DIR / "company_timing.csv", index=False)
    cal_t.to_csv(TABLES_DIR / "company_cut_calibration.csv", index=False)
    df[["bed_id", "variety", "flower_type", "flowers", "est_stems", "est_stems_today", "est_stems_tomorrow",
        "est_stems_later", "cal_est_stems_today", "cal_est_stems_tomorrow", "cal_est_stems_later",
        "factor", "factor_source", "stems", "flags"]].rename(columns={
            "flowers": "flowers_counted", "factor": "flowers_per_stem_used", "stems": "company_stems"}
    ).to_csv(TABLES_DIR / "beds_flowers_stems.csv", index=False)

    ratio = v.flowers / v.stems
    out = {
        "note": "validación indirecta: la empresa reporta tallos, no flores",
        "report_dates": [str(rep.reception_date.min().date()), str(rep.reception_date.max().date())],
        "beds": len(df), "valid_beds": int(df.valid.sum()),
        "excluded": dict(zip(df.video[~df.valid], df["flags"][~df.valid])),
        "flowers_per_stem": {"mean": round(float(ratio.mean()), 2), "min": round(float(ratio.min()), 2),
                             "max": round(float(ratio.max()), 2),
                             "cv_pct": round(float(100 * ratio.std() / ratio.mean()), 1),
                             "by_variety": group_stats(df, "variety").reset_index().to_dict(orient="records"),
                             "by_flower_type": group_stats(df, "flower_type").reset_index().to_dict(orient="records")},
        "stems_leave_one_out": {
            "line": stem_errors(v.loo_stems_line, v.stems),
            "tracking": stem_errors(v.loo_stems_tracking, v.stems),
            "line_without_outlier_63-63A": stem_errors(v[v.bed_id != "63-63A"].loo_stems_line,
                                                       v[v.bed_id != "63-63A"].stems),
            "per_bed": v[["bed_id", "variety", "flower_type", "loo_factor_source", "loo_error_pct_line",
                          "loo_error_pct_tracking"]].to_dict(orient="records")},
        "stems_same_variety_only": same_variety_only(df),
        "factor_grouping_sensitivity_mape_pct": factor_sensitivity(df),
        "cut_calibration": {**cal, "beds": cal_t.drop(columns=["variety"]).to_dict(orient="records")},
        "timing_valid_beds": {
            "pred_today_pct_mean": round(float(v.pred_today_pct.mean()), 1),
            "pred_today_pct_calibrated_mean": round(float(v.cal_today_pct.mean()), 1),
            "company_day0_pct_mean": round(float(v.company_day0_pct.mean()), 1),
            "pred_median_day": v.set_index("bed_id").pred_median_day.to_dict(),
            "company_median_day": v.set_index("bed_id").company_median_day.to_dict()},
    }
    (METRICS_DIR / "company_comparison.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=int))
    fig_ratio(df)
    fig_stems(df)
    fig_timing(tim, df, cum_cal, cal["d_cut_calibrated_cm"], P["cut"]["coin_diameter_cm"])

    pd.set_option("display.width", 200)
    print(df[["bed_id", "variety", "flower_type", "flowers", "flowers_per_stem", "factor_source",
              "loo_error_pct_line", "valid", "flags"]].to_string(index=False))
    print(json.dumps({k: out[k] for k in ["stems_leave_one_out", "factor_grouping_sensitivity_mape_pct"]},
                     indent=1, ensure_ascii=False, default=int))
    print("cut calibration:", {k: cal[k] for k in cal if k != "beds"})


if __name__ == "__main__":
    main()
