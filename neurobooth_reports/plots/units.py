"""
Functions for unit conversions when plotting
"""

from typing import Tuple


INCH_TO_CM = 2.54
CM_TO_INCH = 1 / INCH_TO_CM


def figsize_cm(width: float, height: float) -> Tuple[float, float]:
    """Accepts a figure size in cm and returns it in matplotlib's native inches"""
    return width * CM_TO_INCH, height * CM_TO_INCH


def figsize_mm(width: float, height: float) -> Tuple[float, float]:
    """Accepts a figure size in mm and returns it in matplotlib's native inches"""
    return (width / 10) * CM_TO_INCH, (height / 10) * CM_TO_INCH
