from __future__ import annotations

import struct


def _looks_like_ilda_format_0_header(header: bytes) -> bool:
    """Check whether a 32-byte chunk looks like an ILDA format 0 header.

    Parameters:
        header (bytes): 32-byte header candidate.

    Returns:
        bool: True if the header has the expected magic and format code.
    """
    if len(header) != 32:
        return False

    if header[0:4] != b"ILDA":
        return False

    # This repo writes: "ILDA" + 3 reserved bytes + format code byte.
    if header[4:7] != b"\x00\x00\x00":
        return False

    format_code = header[7]
    return format_code == 0


def _find_next_header_offset(
    data: bytes, *, body_start: int, search_from: int
) -> int | None:
    """Locate the next ILDA header starting from a body offset.

    ILDA point records for format 0 are 8 bytes each, so a valid next header must
    start on an 8-byte boundary relative to ``body_start``.

    Parameters:
        data (bytes): Full ILDA file contents.
        body_start (int): Start offset of the current frame body.
        search_from (int): Offset at which to begin searching.

    Returns:
        int | None: Offset of the next header, or None if not found.
    """
    pos = max(search_from, body_start)

    while True:
        candidate = data.find(b"ILDA", pos)
        if candidate == -1:
            return None

        if (candidate - body_start) % 8 != 0:
            pos = candidate + 1
            continue

        if candidate + 32 <= len(data) and _looks_like_ilda_format_0_header(
            data[candidate : candidate + 32]
        ):
            return candidate

        pos = candidate + 1


def _choose_blanking_bit(statuses: list[int]) -> int:
    """Choose which status bit represents blanking.

    ILDA spec typically uses 0x80 for blanking and 0x40 for "last point".

    This repository's current encoder (see `src/ilda/ilda_3d.py`) uses 0x40 as a
    blanking marker for "pen-up" moves at the start of each polyline and for
    dwell/padding points.

    Parameters:
        statuses (list[int]): Status bytes from point records.

    Returns:
        int: The bit mask to interpret as blanking.
    """
    # If we see any point that has blanking (0x80) without also being marked as
    # last-point (0x40), prefer the spec-conformant blanking bit.
    if any((status & 0x80) and not (status & 0x40) for status in statuses):
        return 0x80

    # If there are no 0x80 blanking points at all, and 0x40 appears on non-final
    # points, treat 0x40 as blanking (matches this repo's writer).
    if not any(status & 0x80 for status in statuses):
        if len(statuses) > 1 and any(status & 0x40 for status in statuses[:-1]):
            return 0x40

    return 0x80


def ilda_to_polylines(
    data: bytes,
    *,
    scale: float | None = None,
    center_x: float = 0.0,
    center_y: float = 0.0,
    invert_y: bool = True,
) -> list[list[tuple[float, float]]]:
    """Convert ILDA file bytes (format 0) into a list of polylines.

    The returned polylines use the project's canonical representation:
    ``list[list[tuple[float, float]]]``.

    Notes:
        - This parser only supports **ILDA format 0** (3D point records).
        - By default, output coordinates are returned in **raw ILDA space**
          (``-32768..32767``) as floats.
        - If `scale` is provided, coordinates are mapped back into the *source*
          coordinate space using the inverse of the encoder transform implemented in
          `src/ilda/ilda_3d.py`.
        - The ILDA "blanking" bit is used to split points into separate polylines.
          The decoder supports both spec-conformant blanking (0x80) and the current
          writer behavior in this repo (0x40).

    Parameters:
        data (bytes): Full contents of an ``.ild`` file.
        scale (float | None): Scale factor used by the encoder. When provided,
            output points are computed as ``(ilda / scale) + center``.
        center_x (float): Center X offset used by the encoder.
        center_y (float): Center Y offset used by the encoder.
        invert_y (bool): Whether the encoder inverted Y when writing ILDA.

    Returns:
        list[list[tuple[float, float]]]: Polylines extracted from the ILDA frames.

    Raises:
        TypeError: If `data` is not bytes-like.
        ValueError: If the data is not a valid ILDA format 0 stream, or if `scale`
            is provided but not strictly positive.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"data must be bytes-like, got {type(data)!r}")

    buf = bytes(data)
    offset = 0

    points: list[tuple[int, int, int]] = []

    while True:
        if offset + 32 > len(buf):
            raise ValueError("Truncated ILDA header")

        header = buf[offset : offset + 32]
        if not _looks_like_ilda_format_0_header(header):
            raise ValueError("Unsupported or invalid ILDA header (expected format 0)")

        num_records = struct.unpack(">H", header[24:26])[0]
        offset += 32

        # EOF marker: header with 0 records.
        if num_records == 0:
            break

        body_start = offset
        expected_body_end = body_start + (num_records * 8)

        if expected_body_end > len(buf):
            raise ValueError("Truncated ILDA body")

        # The writer in this repo can append extra records without updating the
        # header's record count, so try to locate the next header boundary.
        if (
            expected_body_end + 4 <= len(buf)
            and buf[expected_body_end : expected_body_end + 4] == b"ILDA"
        ):
            body_end = expected_body_end
        else:
            next_header = _find_next_header_offset(
                buf,
                body_start=body_start,
                search_from=expected_body_end,
            )
            if next_header is None:
                raise ValueError("Could not find next ILDA header boundary")
            body_end = next_header

        body = buf[body_start:body_end]
        if len(body) % 8 != 0:
            raise ValueError("ILDA format 0 body is not aligned to 8-byte records")

        for rec_offset in range(0, len(body), 8):
            x, y, _z, status, _color = struct.unpack(
                ">hhhBB", body[rec_offset : rec_offset + 8]
            )
            points.append((x, y, status))

        offset = body_end

    statuses = [status for _x, _y, status in points]
    blank_bit = _choose_blanking_bit(statuses)

    polylines: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    if scale is not None and scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    for x, y, status in points:
        if status & blank_bit and current:
            polylines.append(current)
            current = []

        x_f = float(x)
        y_f = float(y)

        if scale is not None:
            x_f = (x_f / scale) + center_x
            if invert_y:
                y_f = (-y_f / scale) + center_y
            else:
                y_f = (y_f / scale) + center_y

        current.append((x_f, y_f))

    if current:
        polylines.append(current)

    return polylines
