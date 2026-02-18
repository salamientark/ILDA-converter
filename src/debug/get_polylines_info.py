def get_polylines_info(polylines: list[list[tuple[float, float]]]) -> tuple[int, int]:
    """
    Get information about the polylines.

    Parameters:
        polylines (list[list[tuple[float, float]]]): List of polylines. Each polyline
            is a list of `(x, y)` points.

    Returns:
        tuple[int, int]: A tuple containing:
            - Total number of points across all polylines.
            - Number of polylines.
    """
    nbr_points = sum(len(polyline) for polyline in polylines)
    nbr_polylines = len(polylines)
    return nbr_points, nbr_polylines
