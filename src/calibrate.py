import os
import cv2
import numpy as np
from typing import List, Union, Dict, Any, Sequence, Tuple

IMG_DIR = r'C:\Users\Yuval\PycharmProjects\3D_vision_final_project\calibration'


# Define the return type with Mat (OpenCV) and ndarray (NumPy)
def get_images(directory: str) -> List[Union[np.ndarray, cv2.Mat]]:
    print(directory)
    images = []
    # Loop over the files in the directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(f"File: {file_path}")
            # Read the image using OpenCV
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
    return images


def detect_ellipses(image: Union[np.ndarray, cv2.Mat]) -> list[dict[str, float | Sequence[float] | Any]]:

    # Define the color range for a tennis ball (yellow-green)
    lower_green = np.array([29, 86, 6])
    upper_green = np.array([64, 255, 255])

    # Create a mask for detecting the tennis ball's color
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)
    # Find contours in the image
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []

    # Loop over the contours
    for contour in contours:
        # Only fit ellipses to contours that have more than 5 points
        if len(contour) >= 5:

            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            (center, (width, height), angle) = ellipse

            # Filter out ellipses based on size and roundness (tennis ball is almost round)
            aspect_ratio = width / height if width > height else height / width
            print(f"aspect ratio: {aspect_ratio}")
            if 0.7 < aspect_ratio < 1.3:  # Adjust size threshold based on your image
                ellipses.append({
                    "center": center,
                    "width": width,
                    "height": height,
                    "angle": angle
                })
            else:
                continue
            # Draw the ellipse on the image for visualization (optional)
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)

            radius = get_ellipse_direction_radius(angle, center, image, width, height)
            print(f"Longest Radius in direction of camera center: {radius}")

    # Display the result
    cv2.imshow("Ellipses Detected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ellipses


def get_ellipse_direction_radius(angle, center, image, width, height):
    # Calculate the direction vector from image center to ellipse center
    image_center = (image.shape[1] // 2, image.shape[0] // 2)
    direction_vector = (image_center[0] - center[0], image_center[1] - center[1])

    # Convert angle to radians for trigonometric functions
    angle_rad = np.deg2rad(angle)

    # Major axis vector (based on the longer dimension of the ellipse)
    major_axis_vector = (np.cos(angle_rad) * width / 2, np.sin(angle_rad) * width / 2)

    # Minor axis vector (based on the shorter dimension of the ellipse)
    minor_axis_vector = (-np.sin(angle_rad) * height / 2, np.cos(angle_rad) * height / 2)

    # Normalize the direction vector
    direction_magnitude = np.linalg.norm(direction_vector)
    normalized_direction_vector = (direction_vector[0] / direction_magnitude, direction_vector[1] / direction_magnitude)

    # Project the normalized direction vector onto the major and minor axes
    projection_major = np.dot(normalized_direction_vector, major_axis_vector) / np.linalg.norm(major_axis_vector)
    projection_minor = np.dot(normalized_direction_vector, minor_axis_vector) / np.linalg.norm(minor_axis_vector)

    # Calculate the final endpoint along the major or minor axis based on the larger projection
    if abs(projection_major) > abs(projection_minor):
        end_point = (
            int(center[0] + major_axis_vector[0]),
            int(center[1] + major_axis_vector[1])
        )
    else:
        end_point = (
            int(center[0] + minor_axis_vector[0]),
            int(center[1] + minor_axis_vector[1])
        )

    # Ensure the line is pointing towards the center of the image
    if (end_point[0] - center[0]) * direction_vector[0] < 0 or (end_point[1] - center[1]) * direction_vector[1] < 0:
        # Flip the endpoint to point toward the image center
        end_point = (
            int(center[0] - (end_point[0] - center[0])),
            int(center[1] - (end_point[1] - center[1]))
        )

    # Draw the line from the ellipse center to the endpoint
    cv2.line(image, (int(center[0]), int(center[1])), end_point, (255, 0, 0), 2)

    # Display the result
    cv2.imshow("Radius towards Image Center", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return the length of the projected radius
    radius_length = np.linalg.norm(np.array(end_point) - np.array(center))
    return radius_length


def select_ball(img: np.ndarray):
    """
    Let the user select the ball on the image.
    The user will select the left, right, top and bottom points.
    :param img: Input image
    :return: List of all selected points
    """
    result = []
    # Create a copy of the image for drawing
    image_draw = img.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            result.append((x, y))
            print((x,y))
            cv2.circle(image_draw, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.imshow("Image", image_draw)

    # Set up window and mouse callback
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", mouse_callback)
    cv2.imshow("Image", img)

    # Wait until the user presses 'q' or until the right number of points is selected
    while True:
        if len(result) == 5:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return np.array(result, dtype=np.int32)


def get_ellipse_from_pts(img, points):
    print("Points selected:", points)

    # Ensure there are enough points to fit an ellipse (at least 5)
    if len(points) < 5:
        raise ValueError("At least 5 points are required to fit an ellipse.")

    ellipse = cv2.fitEllipse(points)
    image_draw = img.copy()
    cv2.ellipse(image_draw, ellipse, (0, 255, 0), 1)
    cv2.imshow("Fitted Ellipse", image_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ellipse


def calibrate_camera():
    images = get_images(IMG_DIR)

    for img in images:
        ellipses = detect_ellipses(img)
        for el in ellipses:
            print(f"Ellipse: Center = {el['center']}, "
                  f"Width = {el['width']}, Height = {el['height']}, "
                  f"Angle = {el['angle']}")

        # points = select_ball(img)
        # ellipse = get_ellipse_from_pts(img, points)
        # print(f"Ellipse: Center = {ellipse[0]}, "
        #       f"Width = {ellipse[1][0]}, Height = {ellipse[1][1]}, "
        #       f"Angle = {ellipse[2]}")


def detect_ball_center_and_ellipse(image_path: str):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the color range for a tennis ball (yellow-green)
    lower_green = np.array([30, 100, 100])
    upper_green = np.array([20, 100, 100])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(hsv, (15, 15), 0)

    # Convert the image to grayscale (for contour detection)
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image (you can adjust the threshold value)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("blurred_image", thresh)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours to find potential ball regions
    for contour in contours:
        if len(contour) >= 5:  # Ellipse fitting requires at least 5 points
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Create a frame (bounding box) around the contour center
                box_size = 100  # Adjust this size based on the image resolution
                top_left = (max(0, cX - box_size), max(0, cY - box_size))
                bottom_right = (min(image.shape[1], cX + box_size), min(image.shape[0], cY + box_size))

                # Crop the region of interest (ROI)
                roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                # Detect edges in the ROI (optional step to help with contour detection)
                edges = cv2.Canny(roi, 50, 150)

                # Find contours in the ROI
                roi_contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Try to fit an ellipse to the largest contour in the ROI
                if len(roi_contours) > 0:
                    largest_contour = max(roi_contours, key=cv2.contourArea)
                    if len(largest_contour) >= 5:
                        ellipse = cv2.fitEllipse(largest_contour)
                        cv2.ellipse(roi, ellipse, (0, 255, 0), 2)

                # Draw the frame around the detected ball on the original image
                cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
                # Draw the center point of the detected ball
                cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

    # Show the final result
    cv2.imshow("Detected Ball and Ellipse", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    calibrate_camera()