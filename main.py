import argparse
import cv2
import potrace

from src.bitmap.preprocessing import (
    binary_img,
    mean_tresh_img,
    gaussian_tresh_img,
    otsu_thresholding,
)


def path_to_svg(path: cv2.typing.MatLike, width: int, height: int) -> None:
    """
    Convert a bitmap path to SVG format.

    Parameters:
        path (cv2.typing.MatLike): The bitmap path to convert.
        width (int): The width of the output SVG.
        height (int): The height of the output SVG.
    """
    parts = []

    # SVG Header
    parts.append(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">')
    parts.append(f'<path d="')

    # Iterate over the curves (shapes) in the path
    for curve in path:
        # 1. Move to the start point of the shape
        start = curve.start_point
        parts.append(f"M {start.x},{start.y}")

        # 2. Iterate over segments in the curve
        for segment in curve:
            if segment.is_corner:
                # Corner segments are composed of two straight lines meeting at 'c'
                # Potrace Corner: Start -> c -> End
                c = segment.c
                end = segment.end_point
                parts.append(f"L {c.x},{c.y} L {end.x},{end.y}")
            else:
                # Bezier segments are cubic curves
                # Potrace Bezier: Start -> c1 -> c2 -> End
                c1 = segment.c1
                c2 = segment.c2
                end = segment.end_point
                parts.append(f"C {c1.x},{c1.y} {c2.x},{c2.y} {end.x},{end.y}")
        
        # 3. Close the shape loop
        parts.append("Z")

    # SVG Footer
    parts.append(f'" stroke="black" fill="none"/>')
    parts.append('</svg>')

    return parts


def run_preprocessing(input: str) -> list[cv2.typing.MatLike]:
    """Run the preprocessing stage on bitmap file.

    Parameters:
        input (str): Path to the input image file.
    """
    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    image_results = []
    image_results.append(binary_img(img))
    image_results.append(mean_tresh_img(img))
    image_results.append(gaussian_tresh_img(img))
    image_results.append(otsu_thresholding(img))

    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)
    image_results.append(otsu_thresholding(gaussian_blur))

    return image_results


def save_img(workspace: str, filename: str, img: cv2.typing.MatLike) -> None:
    """
    Save the processed image to the specified workspace.

    File format is determined by the filename (jpg, pbm, pgm, ppm...)

    Parameters:
        workspace (str): Path to the workspace directory.
        filename (str): Name of the file to save the image as.
        img (cv2.typing.MatLike): The image to be saved.
    """
    if workspace.endswith("/"):
        workspace = workspace[:-1]
    cv2.imwrite(f"{workspace}/{filename}", img)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert image to black and white")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=False, help="Path to output image")
    return parser.parse_args()

def main():
    # Init
    args = parse_args()
    workspace = "data/smiley/preprocessing"
    filenames = [
        "binary_image.pbm",
        "mean_threshold_image.jpg",
        "gaussian_threshold_image.jpg",
        "otsu_threshold_image.jpg",
        "otsu_threshold_gaussian_blur_image.jpg",
    ]

    try:
        # Open image
        images = run_preprocessing(args.input)

        # Create potrace point list
        points = []
        for filename, img in zip(filenames, images):
            # Save result image
            save_img(workspace, filename, img)

            # Create potrace bitmap
            bitmap = potrace.Bitmap(img)

            path = bitmap.trace(
                turdsize=2, # Suppress speckles of up to this size.
                turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY, # How to resolve ambiguities in path direction.
                alphamax=1.0, # Corner threshold parameter.
                opticurve=True, # Whether to use optimized curves.
                opttolerance=0.2, # Curve optimization tolerance.
            )

            raw_svg = path_to_svg(path, img.shape[1], img.shape[0])

            # Save SVG file
            with open(f"{workspace}/{filename.split('.')[0]}.svg", "w") as svg_file:
                svg_file.writelines("\n".join(raw_svg))
                print(f"Saved SVG: {workspace}/{filename.split('.')[0]}.svg")



            # Show the image
            cv2.imshow("Image", img)

            # Wait for a key press and close the image window
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
