# -*- coding: utf-8 -*-
"""
Utilidades internas compartidas por los módulos del paquete.

- ``str_bool``: conversión tolerante a booleano (NaN/None/"no_info" -> False).
- ``normalize_id`` / ``normalize_id_series``: normalización de IDs para que
  ``1``, ``1.0``, ``"1"``, ``"0001"`` y ``" 1 "`` se consideren el mismo predio.
- ``echo``: ``print`` que no falla si la consola no soporta un carácter.
"""

from __future__ import annotations

import re
import sys
import warnings

import pandas as pd

_TRUE_STRINGS = frozenset(("true", "1", "yes", "y", "si", "sí", "t"))
_FLOAT_INT_RE = re.compile(r"^\d+\.0+$")
_DIGITS_RE = re.compile(r"^\d+$")


def str_bool(x) -> bool:
    """Convierte un valor a booleano de forma tolerante.

    Valores faltantes (``None``, ``NaN``, ``pd.NA``) se interpretan como
    ``False``. Esto es importante: ``bool(float("nan"))`` es ``True`` en
    Python, lo que convertiría huecos de un merge en alertas falsas.
    Strings se evalúan contra una lista de verdaderos conocidos; cualquier
    otro string (p. ej. ``"no_info"``) es ``False``.
    """
    if isinstance(x, str):
        return x.strip().lower() in _TRUE_STRINGS
    try:
        if pd.isna(x):
            return False
    except (TypeError, ValueError):
        # pd.isna sobre listas/arrays no devuelve un escalar
        pass
    try:
        return bool(x)
    except Exception:
        return False


def normalize_id(value) -> str:
    """Normaliza un ID a string limpio (sin ``.0``, sin ceros a la izquierda, mayúsculas).

    ``None``/``NaN`` devuelven ``""``.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if s.lower() in ("nan", "none", "<na>"):
        return ""
    if _FLOAT_INT_RE.match(s):
        s = s.split(".", 1)[0]
    if _DIGITS_RE.match(s):
        s = str(int(s))
    return s.upper()


def normalize_id_series(series: pd.Series) -> pd.Series:
    """Aplica :func:`normalize_id` a una Serie, evaluando cada valor único una sola vez."""
    uniques = pd.unique(series)
    mapping = {v: normalize_id(v) for v in uniques}
    return series.map(mapping).astype(str)


def echo(message: str) -> None:
    """Imprime un mensaje de progreso sin fallar por codificación.

    En Windows, cuando la salida se redirige a un archivo o a otro proceso,
    ``sys.stdout`` suele usar cp1252 y ``print`` lanza ``UnicodeEncodeError``
    con emojis. En ese caso los caracteres no representables se reemplazan.
    """
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(message.encode(encoding, errors="replace").decode(encoding))


def warn_deprecated(old: str, new: str, note: str = "") -> None:
    """Emite un DeprecationWarning apuntando al código de quien llama a ``old``."""
    message = f"{old} está obsoleta y se eliminará en una versión futura; use {new}."
    if note:
        message += " " + note
    warnings.warn(message, DeprecationWarning, stacklevel=3)
