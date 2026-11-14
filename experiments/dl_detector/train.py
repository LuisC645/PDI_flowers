"""Entrena un YOLO nano con las pre-etiquetas. Salida: data/dl/runs/flowers/weights/best.pt"""
import argparse

import torch
from ultralytics import YOLO

from common import DL_DIR, IMGSZ, RUNS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(args.model)
    model.train(data=str(DL_DIR / "data.yaml"), imgsz=IMGSZ, epochs=args.epochs, batch=16,
                device=device, project=str(RUNS_DIR), name="flowers", exist_ok=True,
                max_det=600, fliplr=0.5, flipud=0.5, mosaic=0.5, hsv_h=0.0, hsv_s=0.2, hsv_v=0.2,
                degrees=0, scale=0.2, plots=True, workers=4, seed=0)
    # hsv_h = 0: el tono distingue la flor abierta (naranja) del botón (amarillo-verdoso)


if __name__ == "__main__":
    main()
