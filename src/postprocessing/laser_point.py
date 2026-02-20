"""
LaserPoint dataclass representing a single point in a laser path.
"""

from __future__ import annotations

from dataclasses import dataclass


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

    def __post_init__(self) -> None:
        """Validate field ranges."""
        for name, val in [("r", self.r), ("g", self.g), ("b", self.b)]:
            if not 0 <= val <= 255:
                raise ValueError(f"{name} must be 0–255, got {val}")
        if self.status not in (0, 1):
            raise ValueError(f"status must be 0 or 1, got {self.status}")

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
