import cv2

def main():
    # Read the image
    img = cv2.imread('data/image_1.jpg', cv2.IMREAD_COLOR)

    # Show the image
    cv2.imshow('Image', img)

    # Wait for a key press and close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
