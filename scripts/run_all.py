"""Corre el pipeline completo en el orden correcto y se detiene en el primer error.

    estandarizar → Bloque A → B1-B3 → calibrar color → B4-B5 → comparación con la empresa

La comparación con la empresa solo corre si están sus datos en data/ground_truth/company/.
Cada paso salta lo que ya está hecho cuando el script lo permite (por ejemplo, los videos ya
estandarizados).

Uso:  python scripts/run_all.py [--video 63_40A] [--from block_b] [--ignore-gate]
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.paths import COMPANY_DIR, load_metadata  # noqa: E402

STAGES = ["standardize", "block_a", "block_b", "calibrate", "block_b_cut", "compare"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="un solo video (B4-B5 y la comparación usan todas las camas)")
    ap.add_argument("--from", dest="start", choices=STAGES, default=STAGES[0], help="empezar en este paso")
    ap.add_argument("--ignore-gate", action="store_true", help="correr el Bloque B aunque A7 no pase")
    args = ap.parse_args()
    load_metadata()   # valida data/videos.csv antes de gastar tiempo
    video = ["--video", args.video] if args.video else []
    gate = ["--ignore-gate"] if args.ignore_gate else []
    commands = {
        "standardize": ["scripts/standardize_videos.py"],
        "block_a": ["scripts/block_a.py", *video],
        "block_b": ["scripts/block_b.py", "--steps", "b12", "b3", *video, *gate],
        "calibrate": ["scripts/calibrate_colors.py"],
        "block_b_cut": ["scripts/block_b.py", "--steps", "b4", "b5", *video, *gate],
        "compare": ["scripts/compare_company.py"],
    }
    for stage in STAGES[STAGES.index(args.start):]:
        if stage == "compare" and not (COMPANY_DIR / "reception_report.csv").exists():
            print("== compare: skipped (no company data in data/ground_truth/company/)")
            continue
        print(f"== {stage}: {' '.join(commands[stage])}", flush=True)
        if subprocess.run([sys.executable, *commands[stage]], cwd=ROOT).returncode != 0:
            raise SystemExit(f"Stopped at '{stage}'. Fix it and resume with --from {stage}")
    print("== done")


if __name__ == "__main__":
    main()
