"""Shared explicit boolean CLI parsing (including --student_thinking False)."""

import argparse


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected True or False")
