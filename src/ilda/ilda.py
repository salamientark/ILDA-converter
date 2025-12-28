import potrace
import struct


def ilda_header(
    num_points: int,
    frame_name: str = "Frame000",
    company_name: str = "ILDA",
    frame_num: int = 0,
    total_frames: int = 1,
) -> bytes:
    """
    Create ILDA Format 1 header.

    Parameters:
        num_points (int): Number of coordinate points in the frame.
        frame_name (str): Name of the frame (max 8 chars, will be truncated).
        company_name (str): Company name (max 8 chars, will be truncated).
        frame_num (int): Current frame number (0-indexed).
        total_frames (int): Total number of frames in the file.

    Returns:
        bytes: 32-byte ILDA Format 1 header.
    """
    # Magic bytes (4 bytes)
    header = b"ILDA"

    # Format code (3 bytes) - Format 1 = 2D coordinates
    header += b"\x00\x01\x00"

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

    # Scanner head (1 byte)
    header += b"\x00"

    # Reserved (2 bytes)
    header += b"\x00\x00"

    return header


def ilda_body(path: potrace.Path) -> bytes:
    """
    Convert potrace path to ILDA Format 1 body (point records).

    Extracts points from the path, automatically scales coordinates to fit
    ILDA range (-32768 to 32767), and applies blanking between curves.

    Parameters:
        path (potrace.Path): The potrace path to convert.

    Returns:
        bytes: Binary point records (6 bytes per point).

    Raises:
        ValueError: If path has no curves.
    """
    # Validate path has curves
    if not path.curves or len(path.curves) == 0:
        raise ValueError("Path has no curves - cannot convert empty path to ILDA")

    # Extract all points from all curves
    all_points = []
    curve_point_lists = []  # Track which points belong to which curve

    for curve in path:
        # Get decomposition points (list of Point objects with x, y attributes)
        vertices = curve.decomposition_points
        # Convert to list of tuples for consistency
        vertices_list = [(v.x, v.y) for v in vertices]
        curve_point_lists.append(vertices_list)
        all_points.extend(vertices_list)

    # Find bounds for auto-scaling
    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Calculate scale to fit ILDA range (-32768 to 32767 = 65535 range)
    x_range = max_x - min_x
    y_range = max_y - min_y

    # Use uniform scaling to preserve aspect ratio
    if x_range > 0 and y_range > 0:
        scale = min(65535 / x_range, 65535 / y_range) * 0.9  # 0.9 for safety margin
    elif x_range > 0:
        scale = 65535 / x_range * 0.9
    elif y_range > 0:
        scale = 65535 / y_range * 0.9
    else:
        scale = 1.0  # Single point or all same coordinates

    # Calculate centers for offset
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Build point records
    body = b""
    total_points = len(all_points)
    point_idx = 0

    for curve_idx, vertices in enumerate(curve_point_lists):
        for vert_idx, (x, y) in enumerate(vertices):
            # Scale and center coordinates
            ilda_x = int((x - center_x) * scale)
            ilda_y = int((y - center_y) * scale)

            # Clamp to valid ILDA range
            ilda_x = max(-32768, min(32767, ilda_x))
            ilda_y = max(-32768, min(32767, ilda_y))

            # Determine status byte
            status = 0x00  # Default: laser on, drawing

            # First point of each curve after the first = blanked (laser off)
            if curve_idx > 0 and vert_idx == 0:
                status |= 0x80  # Set blanking bit

            # Last point in entire frame gets last-point flag
            if point_idx == total_points - 1:
                status |= 0x40  # Set last point bit

            # Color index (hardcoded to 0 for monochrome white)
            color = 0

            # Pack point record: X (2 bytes), Y (2 bytes), status (1 byte), color (1 byte)
            point_record = struct.pack(">hhBB", ilda_x, ilda_y, status, color)
            body += point_record

            point_idx += 1

    return body


def ilda_footer() -> bytes:
    """
    Create ILDA footer (end-of-file marker).

    The ILDA EOF marker is a header with 0 points.

    Returns:
        bytes: 32-byte ILDA footer.
    """
    return ilda_header(num_points=0, frame_name="", company_name="")


def path_to_ilda(path: potrace.Path) -> list[bytes]:
    """
    Convert a potrace path to ILDA format.

    Parameters:
        path (potrace.Path): The potrace path to convert.

    Returns:
        list[bytes]: The ILDA representation as [header, body, footer].

    Raises:
        ValueError: If path has no curves.
    """
    # Generate body first (may raise ValueError)
    body = ilda_body(path)

    # Calculate number of points from body length
    # Each point = 6 bytes (X:2 + Y:2 + status:1 + color:1)
    num_points = len(body) // 6

    # Generate header with correct point count
    header = ilda_header(num_points=num_points)

    # Generate footer (EOF marker)
    footer = ilda_footer()

    # Return as list of byte chunks
    return [header, body, footer]
