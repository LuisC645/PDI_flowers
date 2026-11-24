"""Importa los datos de la empresa a data/ground_truth/company/ (carpeta ignorada por git).

Entrada (carpeta de la empresa, por defecto ../instr/data_oficial/):
    Reporte mov. recepcion*.xls   tallos recibidos en poscosecha por cama y por día
    videos.csv                    variedad y fechas de corte por cama (fechas DD/MM/AAAA)
Salida:
    reception_report.csv   bed_id, block, bed, flower_type, variety, week, plant_date, reception_date, stems
    beds_company.csv       video_id, variety, cut_start_date, cut_end_date, note

Leer el .xls necesita xlrd:  pip install xlrd
Uso:  python scripts/import_company.py [--src ../instr/data_oficial]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.paths import COMPANY_DIR, load_metadata  # noqa: E402

# Correcciones a la tabla oficial verificadas en el video (por ejemplo 63-38B: el reporte trae dos
# variedades y el video muestra la blanca; decisión 1 en config/decisions.md). Van en un archivo
# fuera de git, porque contienen datos de la empresa: video_id, column, value
OVERRIDES = COMPANY_DIR / "overrides.csv"
REPORT_COLUMNS = ["flower_type", "block", "bed", "week", "plant_date", "reception_date", "variety", "stems"]


def import_report(src):
    files = sorted(src.glob("Reporte mov. recepcion*.xls*"))
    if not files:
        raise SystemExit(f"No 'Reporte mov. recepcion*.xls' in {src}")
    try:
        d = pd.read_excel(files[-1])
    except ImportError as e:
        raise SystemExit(f"{e}\nInstall the Excel reader: pip install xlrd")
    if d.shape[1] != len(REPORT_COLUMNS):
        raise SystemExit(f"{files[-1].name}: expected {len(REPORT_COLUMNS)} columns "
                         f"(Flor, Bloque, Cama Id, Semana, Siembra, Recepción, Variedad, Tallos), got {list(d.columns)}")
    d.columns = REPORT_COLUMNS
    d["bed_id"] = d.block.astype(str) + "-" + d.bed.astype(str).str.replace(" ", "")
    for c in ["plant_date", "reception_date"]:
        d[c] = pd.to_datetime(d[c], format="%Y/%m/%d").dt.date
    d = d[["bed_id", "block", "bed", "flower_type", "variety", "week", "plant_date", "reception_date", "stems"]]
    d.to_csv(COMPANY_DIR / "reception_report.csv", index=False)
    print(f"{files[-1].name}: {len(d)} rows, {d.reception_date.min()} to {d.reception_date.max()}")


def import_beds(src):
    path = src / "videos.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    off = pd.read_csv(path, dtype=str).fillna("")
    for c in ["cut_start_date", "cut_end_date"]:
        off[c] = pd.to_datetime(off[c], format="%d/%m/%Y", errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    meta = load_metadata()
    beds = meta[["video_id", "bed_id"]].merge(off[["bed_id", "variety", "cut_start_date", "cut_end_date"]],
                                              on="bed_id", how="left").fillna("")
    if OVERRIDES.exists():
        for o in pd.read_csv(OVERRIDES, dtype=str).itertuples():
            beds.loc[beds.video_id == o.video_id, o.column] = o.value
            print(f"override: {o.video_id}.{o.column} = {o.value}")
    out = COMPANY_DIR / "beds_company.csv"
    if out.exists():   # conservar las notas escritas a mano
        notes = pd.read_csv(out, dtype=str).fillna("")[["video_id", "note"]]
        beds = beds.merge(notes, on="video_id", how="left").fillna("")
    else:
        beds["note"] = ""
    missing = beds.video_id[beds.variety == ""].tolist()
    if missing:
        print(f"WARNING: no company row for {missing}")
    beds.drop(columns="bed_id").to_csv(out, index=False)
    print(f"videos.csv: {len(beds)} beds -> {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT.parent / "instr" / "data_oficial"))
    src = Path(ap.parse_args().src)
    COMPANY_DIR.mkdir(parents=True, exist_ok=True)
    import_report(src)
    import_beds(src)


if __name__ == "__main__":
    main()
