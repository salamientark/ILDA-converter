import argparse
import cv2

from src.bitmap.preprocessing import (
    binary_img,
    mean_tresh_img,
    gaussian_tresh_img,
    otsu_thresholding,
)


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
    image_results.append(otsu_thresholding(img))

    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)
    image_results.append(otsu_thresholding(gaussian_blur))

    return image_results


def save_img(workspace: str, filename: str, img: cv2.Matlike) -> None:
    """
    Save the processed image to the specified workspace.

    Parameters:
        workspace (str): Path to the workspace directory.
        filename (str): Name of the file to save the image as.
        img (cv2.Matlike): The image to be saved.
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
    workspace = "data/smiley/preprocessing/"
    args = parse_args()

    try:
        # Open image
        images = run_preprocessing(args.input)

        for i in range(len(images)):
            # Save result image
            if i == 0:
                save_img(workspace, "binary_image.jpg", images[i])
            if i == 1:
                save_img(workspace, "mean_threshold_image.jpg", images[i])
            if i == 2:
                save_img(workspace, "gaussian_threshold_image.jpg", images[i])
            if i == 3:
                save_img(workspace, "otsu_threshold_image.jpg", images[i])
            if i == 4:
                save_img(workspace, "otsu_threshold_gaussian_blur_image.jpg", images[i])

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
