"""Structured, multi-output blackbox function support.

A blackbox function checked by fiddy does not have to return a single
plain array. It may instead return a dict of named arrays -- e.g. a bundle
of ``{"x": ..., "y": ..., "sigmay": ..., "llh": ...}`` -- and this module
flattens that into one vector so a single finite-difference sweep
produces derivatives for every named output at once, with no extra
function evaluations, then restores the named structure afterward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["OutputSchema", "flatten_output"]


@dataclass
class OutputSchema:
    """Describes how a flat array bundles one or more named output arrays.

    ``is_structured`` is ``False`` when the original output was a single
    plain array (not a dict) -- :meth:`unbundle` then returns a plain
    array again, rather than a single-entry dict.
    """

    names: list[str]
    shapes: dict[str, tuple[int, ...]]
    slices: dict[str, slice]
    is_structured: bool

    @property
    def size(self) -> int:
        """Total length of the flat array this schema describes."""
        return sum(
            int(np.prod(shape, dtype=int)) if shape else 1
            for shape in self.shapes.values()
        )

    def unbundle(self, flat: np.ndarray) -> dict[str, np.ndarray] | np.ndarray:
        """Restore the named (or plain) output structure from a flat array.

        ``flat`` need not be the original function output -- it is
        typically a *derivative* with the same flat layout (e.g. one row
        of a Jacobian), which is exactly the point: the schema captured
        from the raw output also describes how to unbundle anything else
        that shares its layout.

        :param flat: A flat array sharing this schema's layout.
        :return: The named outputs as a dict, or a plain array if
            :attr:`is_structured` is `False`.
        """
        flat = np.asarray(flat)
        result = {
            name: flat[self.slices[name]].reshape(self.shapes[name])
            for name in self.names
        }
        if not self.is_structured:
            return result[self.names[0]]
        return result


def flatten_output(output: Any) -> tuple[np.ndarray, OutputSchema]:
    """Flatten a plain array or a dict of named arrays into one flat vector.

    A plain array-like output becomes a single unnamed component
    (``schema.is_structured is False``). A dict of named array-likes is
    concatenated in insertion order into one vector; :class:`OutputSchema`
    records each name's original shape and its slice of the flat vector so
    the structure can be restored later via :meth:`OutputSchema.unbundle`.

    :param output: A plain array-like, or a dict of named array-likes.
    :return: A tuple of the flat array and the schema describing it.
    """
    if isinstance(output, dict):
        names = list(output.keys())
        arrays = {name: np.asarray(output[name]) for name in names}
        is_structured = True
    else:
        names = ["_"]
        arrays = {"_": np.asarray(output)}
        is_structured = False

    shapes: dict[str, tuple[int, ...]] = {}
    slices: dict[str, slice] = {}
    parts: list[np.ndarray] = []
    offset = 0
    for name in names:
        array = arrays[name]
        shapes[name] = array.shape
        flat_part = array.reshape(-1)
        slices[name] = slice(offset, offset + flat_part.size)
        parts.append(flat_part)
        offset += flat_part.size

    flat = np.concatenate(parts) if parts else np.array([])
    schema = OutputSchema(
        names=names, shapes=shapes, slices=slices, is_structured=is_structured
    )
    return flat, schema
