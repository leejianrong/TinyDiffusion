"""
helpers.py - Small utility functions used in the 
UNet, GaussianDiffusion, and Trainer classes.
"""

from __future__ import annotations

import math
from typing import Callable, Generator, Iterable, List


# ──────────────────────────────────────────────────────────────────────────────
# General-purpose helpers
# ──────────────────────────────────────────────────────────────────────────────


def exists(value) -> bool:
    """Return ``True`` iff *value* is **not** ``None``."""
    return value is not None


def default(value, fallback):
    """
    If *value* is ``None`` return *fallback*.

    *fallback* may be a value **or** a 0-ary callable to be invoked lazily.
    """
    if exists(value):
        return value
    return fallback() if callable(fallback) else fallback


def cast_tuple(item, length: int = 1) -> Tuple:
    """Broadcast *item* to a tuple of length *length* (no copy if already tuple)."""
    return item if isinstance(item, tuple) else (item,) * length


def divisible_by(numerator: int, denominator: int) -> bool:
    """``True`` iff *numerator* is an integer multiple of *denominator*."""
    return numerator % denominator == 0


identity: Callable = lambda x, *_, **__: x  # quick alias for functional pipes


def endless_cycle(loader: Iterable) -> Generator:
    """
    Yield elements from *loader* **forever**.

    Useful for training loops where the dataloader is shorter than the
    optimization horizon.
    """
    while True:
        yield from loader


def has_integer_sqrt(number: int) -> bool:
    """Return ``True`` if *number* is a perfect square."""
    root = int(math.isqrt(number))
    return root * root == number


def split_into_groups(total: int, group_size: int) -> List[int]:
    """
    Break *total* samples into a list of subgroup sizes.

    Example
    -------
    >>> split_into_groups(10, 4)  # → [4, 4, 2]
    """
    full_groups, remainder = divmod(total, group_size)
    sizes = [group_size] * full_groups
    if remainder:
        sizes.append(remainder)
    return sizes


# ──────────────────────────────────────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────────────────────────────────────


def convert_image_mode(img, mode: str):
    """
    Convert a PIL image to *mode* if necessary; otherwise return unchanged.
    """
    return img.convert(mode) if img.mode != mode else img


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers (tensor inputs expected in **[0, 1]** range)
# ──────────────────────────────────────────────────────────────────────────────

normalize_to_neg_one_to_one = lambda t: t * 2.0 - 1.0
unnormalize_to_zero_to_one = lambda t: (t + 1.0) * 0.5

