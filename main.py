"""
ILDA: Image processing pipeline for bitmap to vector conversion.

Main entry point for the ILDA application, handling command-line argument parsing
and pipeline execution.
"""

import argparse

from src.pipeline.orchestrator import run_pipeline


def parse_args():
    """Parse command line arguments for the ILDA image processing pipeline."""
    parser = argparse.ArgumentParser(
        description="ILDA: Image processing pipeline for bitmap to vector conversion"
    )
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=False, help="Path to output image")
    return parser.parse_args()


def main():
    """
    Main entry point for the ILDA image processing pipeline.

    Parses command line arguments and executes the full processing pipeline
    including preprocessing and vectorization stages.
    """
    args = parse_args()

    try:
        run_pipeline(args.input)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
