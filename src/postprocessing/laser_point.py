"""
LaserPoint dataclass representing a single point in a laser path.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.logger.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class LaserPoint:
    """A single point in a laser display path.

    Attributes:
        x: Horizontal coordinate.
        y: Vertical coordinate.
        r: Red channel (0–255).
        g: Green channel (0–255).
        b: Blue channel (0–255).
        status: Beam state — 0 = on/visible, 1 = off/blanking.
    """

    x: float
    y: float
    r: int = 0
    g: int = 0
    b: int = 0
    status: int = 0

    @classmethod
    def from_xy(
        cls,
        x: float,
        y: float,
        *,
        r: int = 0,
        g: int = 0,
        b: int = 0,
        status: int = 0,
    ) -> LaserPoint:
        """Create a LaserPoint from coordinates and optional colour/status.

        Args:
            x: Horizontal coordinate.
            y: Vertical coordinate.
            r: Red channel (0–255).
            g: Green channel (0–255).
            b: Blue channel (0–255).
            status: Beam state — 0 = on/visible, 1 = off/blanking.

        Returns:
            A new LaserPoint instance.
        """
        return cls(x=x, y=y, r=r, g=g, b=b, status=status)

    @property
    def is_blanking(self) -> bool:
        """Return True when the beam is off (blanking travel point)."""
        return self.status == 1
