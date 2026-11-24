import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from ganabosques_risk_package._utils import echo, normalize_id, normalize_id_series, str_bool


class TestEcho(unittest.TestCase):

    def test_echo_survives_cp1252_console(self):
        # Simula stdout de Windows redirigido a archivo (cp1252, sin emojis)
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
        with contextlib.redirect_stdout(stream):
            echo("📊 Predios con alerta: 3 — métricas")
        stream.flush()
        self.assertEqual(raw.getvalue().decode("cp1252").strip(), "? Predios con alerta: 3 — métricas")


class TestStrBool(unittest.TestCase):

    def test_missing_values_are_false(self):
        # bool(float("nan")) es True en Python; aquí debe ser False.
        for v in (np.nan, float("nan"), None, pd.NA, pd.NaT):
            self.assertFalse(str_bool(v), repr(v))

    def test_true_values(self):
        for v in (True, np.True_, 1, 1.0, "True", " true ", "1", "yes", "si", "sí", "T"):
            self.assertTrue(str_bool(v), repr(v))

    def test_false_values(self):
        for v in (False, np.False_, 0, 0.0, "False", "0", "no", "", "no_info"):
            self.assertFalse(str_bool(v), repr(v))


class TestNormalizeId(unittest.TestCase):

    def test_equivalent_forms(self):
        for v in (1, 1.0, "1", "1.0", "0001", " 1 ", np.int64(1), np.float64(1.0)):
            self.assertEqual(normalize_id(v), "1", repr(v))

    def test_alphanumeric_uppercased(self):
        self.assertEqual(normalize_id(" ab-12 "), "AB-12")
        self.assertEqual(normalize_id("007abc"), "007ABC")  # solo se quitan ceros en IDs numéricos

    def test_missing_is_empty(self):
        for v in (None, np.nan, pd.NA, "nan", "None"):
            self.assertEqual(normalize_id(v), "", repr(v))

    def test_series(self):
        s = pd.Series([1.0, np.nan, "0012", " ab ", None, 7], dtype=object)
        self.assertEqual(normalize_id_series(s).tolist(), ["1", "", "12", "AB", "", "7"])


if __name__ == "__main__":
    unittest.main()
