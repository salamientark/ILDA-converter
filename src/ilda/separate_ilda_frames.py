def separate_ilda_frames(data: bytes) -> list[list[list[tuple[float, float]]]]:
    """
    Separate ILDA file bytes into frames of polylines.

    Parameters:
        data (bytes): Full contents of an ``.ild`` file.
    Returns:
        list[list[list[tuple[float, float]]]]: Frames of polylines extracted from the ILDA file.
    """
    frames = []
    target = b"ILDA\x00\x00\x00"
    index = 0
    next_index = data.find(target, index + 1)
    while next_index != -1:
        frames.append(data[index:next_index])
        index = next_index
        next_index = data.find(target, index + 1)
    return frames
