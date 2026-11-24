"""Pruebas mínimas de las piezas que deciden números del informe.

Correr:  python -m unittest discover tests
No necesitan videos ni datos de la empresa: usan tablas pequeñas hechas a mano.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.counting import classify_maturity, color_index  # noqa: E402
from src.common.paths import load_metadata, load_params  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CUT = {"coin_diameter_cm": 2.0, "growth_per_day": 0.10, "max_days": 7}


class TestMaturity(unittest.TestCase):
    def flowers(self, d, sat):
        return pd.DataFrame({"d_cm": d, "saturation": sat, "value": [200] * len(d)})

    def test_stages(self):
        f = classify_maturity(self.flowers([2.5, 1.9, 1.0, 2.5], [200, 200, 200, 50]), "orange", 100, CUT)
        # grande y madura → hoy; le falta < 1 día → mañana; muy pequeña → otros; grande y pálida → mañana
        self.assertEqual(f.stage.tolist(), ["today", "tomorrow", "later", "tomorrow"])

    def test_days_to_size(self):
        f = classify_maturity(self.flowers([2.0 / 1.1 ** 3, 0.1], [200, 200]), "orange", 100, CUT)
        self.assertEqual(f.days_to_size.iloc[0], 3)
        self.assertEqual(f.days_to_size.iloc[1], CUT["max_days"])   # tope

    def test_min_cut_ratio_moves_threshold(self):
        f = self.flowers([2.5], [200])
        self.assertEqual(classify_maturity(f, "orange", 100, CUT).stage.iat[0], "today")
        self.assertEqual(classify_maturity(f, "orange", 100, CUT, min_cut_ratio=1.5).stage.iat[0], "later")

    def test_white_uses_whiteness(self):
        f = pd.DataFrame({"saturation": [10, 80], "value": [230, 230]})
        self.assertEqual(color_index(f, "white").tolist(), [220, 150])


class TestCompanyComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cc = load_script("compare_company")

    def df(self):
        return pd.DataFrame({
            "video": list("abcde"), "variety": ["R", "R", "M", "N", "N"],
            "flower_type": ["D", "D", "D", "I", "I"], "flowers": [400, 600, 500, 900, 700],
            "flowers_other_method": [400, 600, 500, 900, 700], "stems": [100, 100, 100, 100, 100],
            "valid": [True, True, True, True, False]})

    def test_weighted_median(self):
        self.assertEqual(self.cc.weighted_median([0, 1, 3], [10, 10, 100]), 3)
        self.assertEqual(self.cc.weighted_median([0, 1, 3], [100, 10, 10]), 0)

    def test_factor_hierarchy(self):
        df = self.df()
        row = df.iloc[0].to_dict()
        k, src, n = self.cc.flowers_per_stem_factor(df, "flowers", row, skip="a")
        self.assertEqual((k, src, n), (6.0, "variety", 1))           # solo "b" (misma variedad)
        row = df.iloc[2].to_dict()
        k, src, n = self.cc.flowers_per_stem_factor(df, "flowers", row, skip="c")
        self.assertEqual((k, src, n), (5.0, "flower_type", 2))       # a y b (mismo tipo)
        row = df.iloc[3].to_dict()
        k, src, n = self.cc.flowers_per_stem_factor(df, "flowers", row, skip="d")
        self.assertEqual(src, "all")                                 # "e" no es válida
        self.assertAlmostEqual(k, 5.0)

    def test_stem_errors(self):
        e = self.cc.stem_errors([110, 90, 100], [100, 100, 100])
        self.assertEqual(e["mape_pct"], 6.7)
        self.assertEqual(e["bias_pct"], 0.0)
        self.assertEqual(e["within_10_pct"], 3)


class TestMetadata(unittest.TestCase):
    def test_videos_csv_is_valid(self):
        meta = load_metadata(include_excluded=True)
        self.assertTrue(meta.video_id.is_unique)
        profiles = set(load_params()["profiles"])
        self.assertTrue(set(meta.flower_color[meta.flower_color != ""]) <= profiles)

    def test_no_company_columns_in_git_metadata(self):
        cols = set(pd.read_csv(ROOT / "data" / "videos.csv", nrows=0).columns)
        self.assertFalse(cols & {"variety", "cut_start_date", "cut_end_date"})


if __name__ == "__main__":
    unittest.main()
