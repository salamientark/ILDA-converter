import argparse
import cv2

from src.bitmap.preprocessing import binary_img, mean_tresh_img, gaussian_tresh_img


def run_preprocessing(input: str) -> list[cv2.Matlike]:
    """Run the preprocessing stage on bitmap file.

    Parameters:
        input (str): Path to the input image file.
    """
    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    image_results = []
    image_results.append(binary_img(img))
    image_results.append(mean_tresh_img(img))
    image_results.append(gaussian_tresh_img(img))
    return image_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert image to black and white")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=False, help="Path to output image")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        # Open image
        images = run_preprocessing(args.input)

        for i in range(len(images)):
            # Save result image
            if i == 0:
                cv2.imwrite("data/smiley/preprocessing/binary_image.jpg", images[i])
            if i == 1:
                cv2.imwrite(
                    "data/smiley/preprocessing/mean_threshold_image.jpg", images[i]
                )
            if i == 2:
                cv2.imwrite(
                    "data/smiley/preprocessing/gaussian_threshold_image.jpg", images[i]
                )

            # Show the image
            cv2.imshow("Image", images[i])

            # Wait for a key press and close the image window
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
