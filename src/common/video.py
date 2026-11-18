"""Lectura y escritura de video.

Los videos originales (pueden venir en HDR y con etiqueta de rotación) se leen con ffmpeg
(read_ffmpeg). Los videos ya estandarizados (SDR, sin rotación) se leen con OpenCV.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

FFMPEG = shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")
FFPROBE = shutil.which("ffprobe") or str(Path(sys.executable).parent / "ffprobe")
HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}  # HLG y PQ
# HDR -> SDR BT.709 con mapeo de tonos (libplacebo, usa la GPU vía Vulkan/MoltenVK)
TONEMAP = ("libplacebo=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv:"
           "tonemapping=bt.2390:format=yuv420p")
TRANSPOSE = {0: None, 90: "transpose=1", 180: "transpose=1,transpose=1", 270: "transpose=2"}


def get_video_info(video_path):
    """Devuelve las propiedades principales de un video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    info = {
        "video": Path(video_path).name,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(fps, 3),
        "n_frames": n_frames,
        "duration_s": round(n_frames / fps, 2),
        "size_mb": round(Path(video_path).stat().st_size / 1e6, 1),
    }
    cap.release()
    return info


def probe_stream(video_path):
    """Datos del video que importan para leerlo bien: tamaño mostrado, HDR, rotación, fps."""
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,color_transfer,avg_frame_rate,nb_frames:"
                          "stream_side_data=rotation", "-of", "json", str(video_path)],
                         capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    rotation = next((int(d["rotation"]) for d in st.get("side_data_list", []) if "rotation" in d), 0)
    w, h = st["width"], st["height"]
    if rotation % 180:  # ffmpeg y OpenCV aplican la rotación de la etiqueta al leer
        w, h = h, w
    num, den = (int(x) for x in st["avg_frame_rate"].split("/"))
    return {"width": w, "height": h, "rotation_tag": rotation,
            "hdr": st.get("color_transfer") in HDR_TRANSFERS, "fps": num / den}


def frame_times(video_path):
    """Tiempo (s) de cada cuadro en orden de presentación (pts de los paquetes)."""
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "packet=pts_time", "-of", "csv=p=0", str(video_path)],
                         capture_output=True, text=True, check=True).stdout
    return sorted(float(x) for x in out.split() if x not in ("", "N/A"))


def read_ffmpeg(video_path, extra_rotation=0, tonemap=None):
    """Recorre el video con ffmpeg y entrega (índice, tiempo_s, cuadro BGR 8 bits).

    - Aplica la rotación de la etiqueta (como OpenCV) y luego extra_rotation (0/90/180/270,
      sentido horario) para dejar al camarógrafo abajo.
    - Si el video es HDR (o tonemap=True), lo convierte a SDR BT.709 con mapeo de tonos.
    """
    info = probe_stream(video_path)
    tonemap = info["hdr"] if tonemap is None else tonemap
    filters = [f for f in (TONEMAP if tonemap else None, TRANSPOSE[extra_rotation % 360]) if f]
    w, h = info["width"], info["height"]
    if extra_rotation % 180:
        w, h = h, w
    cmd = [FFMPEG, "-v", "error", "-i", str(video_path)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    times = frame_times(video_path)
    env = {**os.environ, "MVK_CONFIG_LOG_LEVEL": "0"}  # silencia MoltenVK
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env, bufsize=w * h * 3 * 4)
    size, idx = w * h * 3, 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            t = times[idx] - times[0] if idx < len(times) else idx / info["fps"]
            yield idx, t, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            idx += 1
    finally:
        if proc.poll() is None:  # se dejó de leer antes del final: cerrar ffmpeg sin ruido
            proc.kill()
        proc.stdout.close()
        proc.wait()
    if proc.returncode not in (0, None) and idx == 0:
        raise RuntimeError(f"ffmpeg could not read {video_path}")


def read_frames(video_path, frame_indices):
    """Lee los cuadros pedidos y devuelve una lista de imágenes BGR."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            raise IOError(f"Cannot read frame {idx}")
        frames.append(frame)
    cap.release()
    return frames


def iter_frames(video_path, with_time=False):
    """Recorre el video en orden y entrega (índice, cuadro) o (índice, tiempo_s, cuadro).

    El tiempo sale del contenedor (CAP_PROP_POS_MSEC), así que es correcto también en
    videos con fps variable (como los del celular).
    """
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if with_time:
            yield idx, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame
        else:
            yield idx, frame
        idx += 1
    cap.release()


def open_writer(out_path, width, height, fps, crf=23):
    """Abre ffmpeg para recibir cuadros BGR por stdin y codificarlos en H.264 sin audio.

    OpenCV decodifica con la matriz BT.601 y ffmpeg convierte BGR -> YUV con BT.601, así
    que el video se etiqueta como tal (smpte170m) para que los reproductores muestren el
    mismo color que ve OpenCV.
    """
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
           "-i", "-", "-an", "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
           "-pix_fmt", "yuv420p", "-colorspace", "smpte170m", "-color_primaries", "bt709",
           "-color_trc", "bt709", "-movflags", "+faststart", str(out_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def close_writer(writer, name=""):
    """Cierra el proceso de ffmpeg y falla si terminó con error."""
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError(f"ffmpeg failed {name}")


def make_mosaic(images, labels, n_cols=6, tile_width=270):
    """Une imágenes reducidas en una cuadrícula, con una etiqueta en cada una."""
    tile_height = int(images[0].shape[0] * tile_width / images[0].shape[1])
    tiles = []
    for image, label in zip(images, labels):
        tile = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(tile, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % n_cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + n_cols]) for i in range(0, len(tiles), n_cols)]
    return np.vstack(rows)
