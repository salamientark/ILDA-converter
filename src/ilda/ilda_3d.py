"""
Reference from https://www.ilda.com/resources/StandardsDocs/ILDA_IDTF14_rev011.pdf
"""

import struct


def ilda_header_3d(
    num_points: int,
    frame_name: str = "Frame000",
    company_name: str = "ILDA",
    frame_num: int = 0,
    total_frames: int = 1,
) -> bytes:
    """
    Create ILDA Format 0 header (3D coordinates).

    Parameters:
        num_points (int): Number of coordinate points in the frame.
        frame_name (str): Name of the frame (max 8 chars, will be truncated).
        company_name (str): Company name (max 8 chars, will be truncated).
        frame_num (int): Current frame number (0-indexed).
        total_frames (int): Total number of frames in the file.

    Returns:
        bytes: 32-byte ILDA Format 0 header.
    """
    # Magic bytes (4 bytes)
    header = b"ILDA"

    # Reserved bytes
    header += b"\x00\x00\x00"

    # Format code - Format 0 = 3D coordinates
    header += b"\x00"

    # Frame name (8 bytes) - truncate if longer, pad with nulls if shorter
    frame_name_bytes = frame_name.encode("ascii")[:8]
    header += frame_name_bytes.ljust(8, b"\x00")

    # Company name (8 bytes) - truncate if longer, pad with nulls if shorter
    company_name_bytes = company_name.encode("ascii")[:8]
    header += company_name_bytes.ljust(8, b"\x00")

    # Number of records (2 bytes, big-endian unsigned)
    header += struct.pack(">H", num_points)

    # Frame number (2 bytes, big-endian unsigned)
    header += struct.pack(">H", frame_num)

    # Total frames (2 bytes, big-endian unsigned)
    header += struct.pack(">H", total_frames)

    # Scanner head (2 byte)
    header += b"\x00\x00"

    return header


def ilda_body_3d(
    polylines: list[list[tuple[float, float]]],
    z_value: int = 0,
    invert_y: bool = True,
) -> tuple[bytes, int]:
    """Convert polylines to ILDA Format 0 body (3D point records).

    Extracts points from the polylines, automatically scales X and Y coordinates to fit
    ILDA range (-32768 to 32767), sets Z coordinate to a fixed value, and applies
    blanking between polylines.

    Note:
        OpenCV-style image coordinates have Y increasing *downward*. Most ILDA renderers
        treat +Y as *up*, so by default this function flips the Y axis via ``invert_y=True``.

    Parameters:
        polylines (list[list[tuple[float, float]]]): The polylines. Each polyline is a list of
            (x, y) points.
        z_value (int): Z coordinate value for all points (default 0).
            Must be in range -32768 to 32767.
        invert_y (bool): Whether to invert the Y axis when mapping points to ILDA.

    Returns:
        bytes, int : Binary point records (8 bytes per point) | Number of points

    Raises:
        ValueError: If polylines are empty, any polyline is empty, or z_value is out of range.
    """
    if z_value < -32768 or z_value > 32767:
        raise ValueError(f"z_value must be in range -32768 to 32767, got {z_value}")

    if not polylines:
        raise ValueError("Polylines are empty - cannot convert empty polylines to ILDA")

    all_points: list[tuple[float, float]] = []
    polyline_point_lists: list[list[tuple[float, float]]] = []

    for polyline_idx, polyline in enumerate(polylines):
        if not polyline:
            raise ValueError(
                f"Polyline at index {polyline_idx} is empty - cannot convert empty polyline to ILDA"
            )

        vertices_list = [(float(x), float(y)) for (x, y) in polyline]
        polyline_point_lists.append(vertices_list)
        all_points.extend(vertices_list)

    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    x_range = max_x - min_x
    y_range = max_y - min_y

    if x_range > 0 and y_range > 0:
        scale = min(65535 / x_range, 65535 / y_range) * 0.9
    elif x_range > 0:
        scale = 65535 / x_range * 0.9
    elif y_range > 0:
        scale = 65535 / y_range * 0.9
    else:
        scale = 1.0

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    body = b""
    total_points = len(all_points)
    point_idx = 0

    for polyline_idx, vertices in enumerate(polyline_point_lists):
        last_x: int = 0
        last_y: int = 0

        for vert_idx, (x, y) in enumerate(vertices):
            ilda_x = int((x - center_x) * scale)
            ilda_y = int((y - center_y) * scale)
            if invert_y:
                ilda_y = -ilda_y

            last_x = ilda_x
            last_y = ilda_y

            ilda_x = max(-32768, min(32767, ilda_x))
            ilda_y = max(-32768, min(32767, ilda_y))

            ilda_z = z_value

            status = 0x00
            if vert_idx == 0:
                status |= 0x40

            color = 0
            body += struct.pack(">hhhBB", ilda_x, ilda_y, ilda_z, status, color)

            point_idx += 1

        body += struct.pack(">hhhBB", last_x, last_y, z_value, 0x40, 0)
        body += struct.pack(">hhhBB", last_x, last_y, z_value, 0x40, 0)
        body += struct.pack(">hhhBB", last_x, last_y, z_value, 0x40, 0)
        body += struct.pack(">hhhBB", last_x, last_y, z_value, 0x40, 0)
        total_points += 4
        point_idx += 4
        if point_idx == total_points:
            body += struct.pack(">hhhBB", last_x, last_y, z_value, 0xC0, 0)

    return body, total_points


def ilda_footer_3d() -> bytes:
    """
    Create ILDA Format 0 footer (end-of-file marker).

    The ILDA EOF marker is a header with 0 points.

    Returns:
        bytes: 32-byte ILDA Format 0 footer.
    """
    return ilda_header_3d(num_points=0, frame_name="", company_name="")


def path_to_ilda_3d(
    polylines: list[list[tuple[float, float]]],
    z_value: int = 0,
    invert_y: bool = True,
) -> list[bytes]:
    """Convert polylines to ILDA Format 0 (3D coordinates).

    Parameters:
        polylines (list[list[tuple[float, float]]]): List of polylines. Each polyline is a
            list of `(x, y)` points.
        z_value (int): Z coordinate value for all points (default 0). Must be in range
            -32768 to 32767.
        invert_y (bool): Whether to invert the Y axis when mapping points to ILDA.

    Returns:
        list[bytes]: The ILDA representation as `[header, body, footer]`.

    Raises:
        ValueError: If `polylines` are empty, any polyline is empty, or `z_value` is out of range.
    """
    # Generate body first (may raise ValueError)
    body, num_points = ilda_body_3d(polylines, z_value, invert_y=invert_y)

    # Generate header with correct point count
    header = ilda_header_3d(num_points=num_points)

    # Generate footer (EOF marker)
    footer = ilda_footer_3d()

    # Return as list of byte chunks
    return [header, body, footer]
