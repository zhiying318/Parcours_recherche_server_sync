"""Shared explicit boolean CLI parsing (including --student_thinking False)."""

import argparse
import math


def parse_vocabulary_clip(value: str) -> float | None:
    """Accept 'none' for the unmodified forward KL objective."""
    if value.lower() == "none":
        return None
    try:
        clip = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a positive number or 'none'") from exc
    if not math.isfinite(clip) or clip <= 0:
        raise argparse.ArgumentTypeError("Expected a positive number or 'none'")
    return clip


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected True or False")
