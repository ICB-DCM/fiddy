"""Private formatting helpers shared by :mod:`fiddy.check`'s report builders.

Not part of the public API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _get_printable_value(value) -> str:
    """Round a scalar (or array) to a short, fixed-precision string for
    display in a report.

    :param value: A scalar or array-like value.
    :return: The value rounded to 6 significant digits, as a string.
    """
    array = np.atleast_1d(value)
    if array.size == 1:
        return f"{float(array.reshape(-1)[0]):.6g}"
    return np.array2string(
        array, precision=6, suppress_small=True, threshold=6
    )


def _wide_display():
    """A :func:`pandas.option_context` for rendering full,
    readably-formatted tables.

    Intended for reports that may only be seen as plain-text CI log
    output, where a truncated/wrapped table or a mix of full-precision and
    scientific-notation floats is hard to read.

    :return: A context manager that widens pandas' display options for its
        duration.
    """
    return pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_rows",
        None,
        "display.float_format",
        "{:.6g}".format,
    )
