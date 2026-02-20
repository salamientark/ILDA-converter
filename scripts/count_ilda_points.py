#!/usr/bin/env python3
import struct
import sys
from pathlib import Path


def count_ilda_3d_points(data: bytes) -> list[int]:
    """Return the number of points in each frame of an ILDA file."""
    counts = []
    offset = 0
    while True:
        if offset + 32 > len(data):
            break
        header = data[offset : offset + 32]
        if header[0:4] != b"ILDA":
            break
        num_records = struct.unpack(">H", header[24:26])[0]
        offset += 32
        if num_records == 0:  # EOF marker
            break
        counts.append(num_records)
        offset += num_records * 8
    return counts


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.ild>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    counts = count_ilda_3d_points(path.read_bytes())
    print(counts)


if __name__ == "__main__":
    main()
