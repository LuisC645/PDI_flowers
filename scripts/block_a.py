"""Bloque A: normalización del video (A1-A7).  [BORRADOR]

Pasos (cada uno lee de disco lo que guardó el anterior):
  a1  nitidez y etiqueta nítido/borroso    -> results/tables/frame_quality_<v>.csv
  a23 máscara y bordes de la cama          -> data/normalized/masks/<v>_bed_masks.npz
                                              data/normalized/transforms/<v>_geometry_raw.csv
  a2op zona del camarógrafo                -> data/normalized/masks/<v>_operator_mask.png
  a4  seguimiento y estabilización         -> data/normalized/transforms/<v>_geometry.csv
  a56 normalización geométrica y de color  -> data/normalized/video/<v>.mp4
                                              data/normalized/transforms/<v>_transforms.csv
  a7  control de calidad (puerta al B)     -> results/metrics/qc_block_a_<v>.json + figura
La salida que usa el Bloque B se lee con src/common/normalized.py (NormalizedVideo).
Uso:  python scripts/block_a.py                 (todos los pasos, todos los videos)
      python scripts/block_a.py --steps a4 a56 --video 63-40A
"""
import argparse
import json
import sys
from collections import deque
from itertools import islice
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.paths import (FIGURES_DIR, MASKS_DIR, METRICS_DIR, NORMALIZED_VIDEO_DIR,  # noqa: E402
                              TABLES_DIR, TRANSFORMS_DIR, list_videos, load_params)
from src.common.video import (close_writer, get_video_info, iter_frames,  # noqa: E402
                              make_mosaic, open_writer, read_frames)
from src.normalization.bed import bed_edges, bed_mask, foliage_lab_stats  # noqa: E402
from src.normalization.color import apply_correction, color_corrections  # noqa: E402
from src.common.normalized import NormalizedVideo  # noqa: E402
from src.normalization.geometry import affine_matrix, output_size  # noqa: E402
from src.normalization.operator import operator_mask, static_frequency  # noqa: E402
from src.normalization.sharpness import label_blur  # noqa: E402
from src.normalization.tracking import stabilize  # noqa: E402

STEPS = ["a1", "a23", "a2op", "a4", "a56", "a7"]


def step_a1(video, P):
    """A1: etiqueta cada cuadro conservado como nítido o borroso."""
    idx = pd.read_csv(TABLES_DIR / f"frame_index_{video.stem}.csv")
    cfg = P["sharpness"]
    idx["sharpness_rel"], idx["label"] = label_blur(idx["sharpness"], cfg["window"],
                                                    cfg["rel_threshold"])
    idx.to_csv(TABLES_DIR / f"frame_quality_{video.stem}.csv", index=False)
    blurry = idx[idx.label == "blurry"]
    print(f"A1: {len(blurry)} / {len(idx)} blurry frames")
    if len(blurry):  # muestra visual de los borrosos para revisarlos
        show = blurry.nsmallest(12, "sharpness_rel")
        frames = read_frames(video, show.new_frame)
        labels = [f"#{r.new_frame} rel={r.sharpness_rel:.2f}" for r in show.itertuples()]
        cv2.imwrite(str(FIGURES_DIR / f"a1_blurry_{video.stem}.jpg"), make_mosaic(frames, labels))


def step_a23(video, P):
    """A2 + A3: máscara de la cama y geometría por cuadro (y estadística del follaje para A6)."""
    cfg, sc = P["bed"], P["bed"]["scale"]
    n = get_video_info(video)["n_frames"]
    rows, masks = [], []
    center_guess, score, prior, prior_frame = None, None, None, -10**9
    gap = cfg["parallax_max_gap"]
    frames = (cv2.resize(f, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
              for _, f in iter_frames(video))
    window = deque(islice(frames, gap + 1))  # cuadro actual + los siguientes (paralaje)
    for i in tqdm(range(n), desc="A2/A3"):
        if not window:
            break
        cur = window[0]
        mask, w_par, score = bed_mask(cur, list(window)[1:], cfg, center_guess, score)
        use_prior = prior if i - prior_frame <= cfg["prior_max_age"] else None
        g = bed_edges(mask, cfg, sc, center_guess, seed=i, prior=use_prior)
        fL, fa, fb = foliage_lab_stats(cur, mask, cfg)
        rows.append({"frame": i, **g, "w_parallax": w_par, "mask_frac": (mask > 0).mean(),
                     "foliage_L": fL, "foliage_a": fa, "foliage_b": fb})
        masks.append(np.packbits(mask > 0, axis=1))
        if g["valid"]:
            center_guess, prior, prior_frame = g["center_y"] * sc, g, i
        window.popleft()
        nxt = next(frames, None)
        if nxt is not None:
            window.append(nxt)
    geom = pd.DataFrame(rows)
    geom.round(4).to_csv(TRANSFORMS_DIR / f"{video.stem}_geometry_raw.csv", index=False)
    np.savez_compressed(MASKS_DIR / f"{video.stem}_bed_masks.npz", masks=np.array(masks),
                        scale=sc, width=mask.shape[1])
    print(f"A2/A3: {geom.valid.mean():.1%} valid; fails: "
          f"{geom.fail_reason[~geom.valid].str.split(',').explode().value_counts().to_dict()}")
    overlay_sample(video, geom, P, "a3_edges")


def overlay_sample(video, geom, P, name, n=12):
    """Mosaico con los bordes estimados dibujados sobre cuadros repartidos en el video."""
    pick = np.linspace(0, len(geom) - 1, n).astype(int)
    frames = read_frames(video, pick)
    labels = []
    for fr, i in zip(frames, pick):
        g = geom.iloc[i]
        ok = g.get("valid", g.get("status") != "invalid")
        color = (255, 0, 255) if ok else (0, 0, 255)
        if "top_a" in g and not np.isnan(g["top_a"]):
            w = fr.shape[1]
            for a, b in [(g.top_a, g.top_b), (g.bottom_a, g.bottom_b)]:
                cv2.line(fr, (0, int(b)), (w, int(a * w + b)), color, 8)
        labels.append(f"#{i} {'ok' if ok else 'X'}")
    cv2.imwrite(str(FIGURES_DIR / f"{name}_{video.stem}.jpg"), make_mosaic(frames, labels))


def make_mosaic(frames, labels):
    from src.common.video import make_mosaic
    return make_mosaic(frames, labels)


def step_a2op(video, P):
    """A2 (complemento): zona fija del camarógrafo, por baja variación temporal de los píxeles."""
    cfg, sc = P["operator"], P["bed"]["scale"]
    info = get_video_info(video)
    small = np.stack([cv2.resize(f, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                      for _, f in tqdm(iter_frames(video), total=info["n_frames"], desc="A2 operator")])
    freq = static_frequency(small, cfg["window"], cfg["step"])
    mask = operator_mask(freq, cfg["min_freq"], cfg["extend_up_frac"], cfg["dilate_px"])
    full = cv2.resize(mask, (info["width"], info["height"]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(MASKS_DIR / f"{video.stem}_operator_mask.png"), full)
    print(f"A2 operator: zone covers {(mask > 0).mean():.1%} of the frame")
    pick = np.linspace(0, len(small) - 1, 12).astype(int)
    cnt, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frames = [cv2.drawContours(f, cnt, -1, (255, 0, 255), 10) for f in read_frames(video, pick)]
    cv2.imwrite(str(FIGURES_DIR / f"a2_operator_{video.stem}.jpg"),
                make_mosaic(frames, [f"#{i}" for i in pick]))


def step_a4(video, P):
    """A4: quita atípicos, interpola huecos cortos y suaviza la geometría."""
    raw = pd.read_csv(TRANSFORMS_DIR / f"{video.stem}_geometry_raw.csv")
    geom = stabilize(raw, P["tracking"])
    geom.round(3).to_csv(TRANSFORMS_DIR / f"{video.stem}_geometry.csv", index=False)
    print("A4:", geom.status.value_counts().to_dict())

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, col in zip(axes, ["angle_deg", "center_y", "width_px"]):
        ax.plot(raw.frame, raw[col].where(raw.valid), ".", ms=2, color="0.6", label="medido")
        ax.plot(geom.frame, geom[col], "-", lw=1.2, label="estabilizado")
        ax.set_ylabel(col)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("cuadro")
    fig.suptitle(f"A4 geometría de la cama — {video.stem}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"a4_geometry_{video.stem}.png", dpi=110)
    plt.close(fig)


def step_a56(video, P):
    """A5 + A6: endereza, escala y centra cada cuadro; iguala el color; guarda la matriz."""
    cfg = P["normalize"]
    geom = pd.read_csv(TRANSFORMS_DIR / f"{video.stem}_geometry.csv")
    raw = pd.read_csv(TRANSFORMS_DIR / f"{video.stem}_geometry_raw.csv")
    info = get_video_info(video)
    ok = geom.status != "invalid"
    out_w, out_h, margin = output_size(cfg, info["width"], geom.width_px[ok].median())
    corr = color_corrections(raw, P["color"]["smooth_window"])

    rows = []
    writer = open_writer(NORMALIZED_VIDEO_DIR / f"{video.stem}.mp4", out_w, out_h,
                         info["fps"], cfg["crf"])
    for i, frame in tqdm(iter_frames(video), total=info["n_frames"], desc="A5/A6"):
        g, c = geom.iloc[i], corr.iloc[i]
        if g.status == "invalid":  # se conserva el cuadro (negro) para no romper la secuencia
            out = np.zeros((out_h, out_w, 3), np.uint8)
            M = np.full((2, 3), np.nan)
        else:
            M = affine_matrix(g, cfg, out_w, out_h)
            out = cv2.warpAffine(frame, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
            if P["color"]["enabled"]:
                out = apply_correction(out, c.gain_L, c.off_a, c.off_b)
        writer.stdin.write(out.tobytes())
        rows.append({"frame": i, "status": g.status, **{f"m{r}{k}": M[r, k]
                     for r in range(2) for k in range(3)},
                     "gain_L": c.gain_L, "off_a": c.off_a, "off_b": c.off_b})
    close_writer(writer, video.name)
    tf = pd.DataFrame(rows)
    tf.round(6).to_csv(TRANSFORMS_DIR / f"{video.stem}_transforms.csv", index=False)
    meta = {"out_w": out_w, "out_h": out_h, "margin_px": margin,
            "src_size": [info["width"], info["height"]],
            "bed_width_px": cfg["bed_width_px"], "bed_width_cm": cfg["bed_width_cm"],
            "px_per_cm": cfg["bed_width_px"] / cfg["bed_width_cm"],
            "parallax_k": cfg["parallax_k"]}
    (TRANSFORMS_DIR / f"{video.stem}_normalized_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"A5/A6: {out_w}x{out_h}, bed rows [{margin}, {out_h - margin}), "
          f"{meta['px_per_cm']:.2f} px/cm")


def step_a7(video, P):
    """A7: vuelve a medir la cama en el video normalizado y compara con las metas de params.qc.

    - Geometría: A2+A3 con el mismo método (fusión temporal y prior) sobre el video normalizado.
    - Sacudida lateral: dy entre cuadros consecutivos (correlación de fase en la franja).
    - Color (A6): desviación de la media Lab del follaje entre cuadros, antes y después.
    - Camarógrafo: fracción de la franja que se excluye por su zona.
    - Contaminación de la máscara frente a la referencia a mano: scripts/evaluate_block_a.py.
    """
    cfg, qc = P["bed"], P["qc"]
    nv = NormalizedVideo(video.stem)
    geom = pd.read_csv(TRANSFORMS_DIR / f"{video.stem}_geometry.csv")
    raw = pd.read_csv(TRANSFORMS_DIR / f"{video.stem}_geometry_raw.csv")
    sc = cfg["scale"]
    W, h_out, margin = nv.meta["bed_width_px"], nv.out_h, nv.margin
    cfg_n = {**cfg, "width_frac_range": [0.5 * W / h_out, 1.0]}
    band_small = cv2.resize(nv.band, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    window = None
    rows, prev, prev_gray = [], None, None
    score, prior, prior_frame = None, None, -10**9
    for i, frame in tqdm(iter_frames(nv.video_path), total=len(geom), desc="A7"):
        small = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small[int(margin * sc):int((h_out - margin) * sc)], cv2.COLOR_BGR2GRAY)
        j = i - 1  # se mide el cuadro anterior (el paralaje necesita el siguiente)
        if prev is not None and geom.status[j] != "invalid" and geom.status[i] != "invalid":
            valid = cv2.resize(nv.has_image(j), None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
            mask, _, score = bed_mask(prev, small, cfg_n, prev_score=score, valid=valid)
            use_prior = prior if j - prior_frame <= cfg["prior_max_age"] else None
            g = bed_edges(mask, cfg_n, sc, seed=j, prior=use_prior, valid=valid)
            if g["valid"]:
                prior, prior_frame = g, j
            if window is None:
                window = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(prev_gray.astype(np.float32),
                                                gray.astype(np.float32), window)
            row = {"frame": j, **g, "dx": dx / sc, "dy": dy / sc, "phase_response": resp}
            if j % qc["step"] == 0:
                fL, fa, fb = foliage_lab_stats(prev, band_small, cfg)
                op = nv.operator_in_frame(j)
                row.update(foliage_L=fL, foliage_a=fa, foliage_b=fb,
                           operator_band_frac=(op[nv.band > 0] > 0).mean(),
                           roi_band_frac=(nv.roi(j)[nv.band > 0] > 0).mean())
            rows.append(row)
        else:
            score = None
        prev, prev_gray = small, gray
    m = pd.DataFrame(rows)
    good = m[m.valid]
    lab = ["foliage_L", "foliage_a", "foliage_b"]
    after = m.dropna(subset=lab)
    before = raw.set_index("frame").loc[after.frame, lab]
    res = {
        "frames_checked": int(len(m)), "remeasure_valid_frac": round(len(good) / max(len(m), 1), 3),
        "width_mean_px": round(good.width_px.mean(), 1),
        "width_cv": round(good.width_px.std() / good.width_px.mean(), 4),
        "center_std_frac": round(good.center_y.std() / W, 4),
        "center_offset_px": round(good.center_y.mean() - h_out / 2, 1),
        "angle_std_deg": round(good.angle_deg.std(), 3),
        "angle_mean_deg": round(good.angle_deg.mean(), 3),
        "lateral_jitter_std_px": round(m.dy.std(), 2),
        "lateral_jitter_std_cm": round(m.dy.std() / nv.px_per_cm, 3),
        "foliage_lab_std_before": [round(x, 2) for x in before.std()],
        "foliage_lab_std_after": [round(x, 2) for x in after[lab].std()],
        "operator_band_frac": round(after.operator_band_frac.mean(), 4),
        "roi_band_frac": round(after.roi_band_frac.mean(), 4),
        "invalid_frac": round((geom.status == "invalid").mean(), 4),
        "interpolated_frac": round((geom.status == "interpolated").mean(), 4),
    }
    res["pass"] = {
        "width_cv": bool(res["width_cv"] <= qc["max_width_cv"]),
        "center": bool(res["center_std_frac"] <= qc["max_center_std_frac"]),
        "angle": bool(res["angle_std_deg"] <= qc["max_angle_std_deg"]),
        "lateral_jitter": bool(res["lateral_jitter_std_px"] <= qc["max_jitter_frac"] * W),
        "color_stability": bool(res["foliage_lab_std_after"][0] < res["foliage_lab_std_before"][0]),
        "invalid": bool(res["invalid_frac"] + res["interpolated_frac"] <= qc["max_invalid_frac"]),
    }
    res["pending_manual"] = ["mask_vs_ground_truth (F1.4): python scripts/evaluate_block_a.py"]
    res["gate_open"] = all(res["pass"].values())
    (METRICS_DIR / f"qc_block_a_{video.stem}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    m.round(3).to_csv(TABLES_DIR / f"qc_remeasure_{video.stem}.csv", index=False)
    print("A7:", json.dumps(res, indent=2, ensure_ascii=False))

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for ax, col, ref in zip(axes, ["width_px", "center_y", "angle_deg", "dy"], [W, h_out / 2, 0, 0]):
        ax.plot(m.frame, m[col].where(m.valid) if col != "dy" else m[col], ".", ms=3)
        ax.axhline(ref, color="k", lw=0.8, ls="--")
        ax.set_ylabel(col)
    axes[-1].set_xlabel("cuadro")
    fig.suptitle(f"A7 cama re-medida en el video normalizado — {video.stem}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"a7_qc_{video.stem}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    ap.add_argument("--video", help="nombre del video sin extensión (por defecto, todos)")
    args = ap.parse_args()
    P = load_params()
    for d in (MASKS_DIR, TRANSFORMS_DIR, NORMALIZED_VIDEO_DIR, METRICS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for video in list_videos():
        if args.video and video.stem != args.video:
            continue
        print(f"=== {video.name}")
        for step in args.steps:
            globals()[f"step_{step}"](video, P)


if __name__ == "__main__":
    main()
