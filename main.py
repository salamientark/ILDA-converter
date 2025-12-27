import argparse

from src.pipeline.orchestrator import run_pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert image to black and white")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=False, help="Path to output image")
    return parser.parse_args()


def main():
    # Init
    args = parse_args()

    try:
        run_pipeline(args.input)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
