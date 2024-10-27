import os
import cv2
import numpy as np
from typing import List, Union, Dict, Any, Sequence, Tuple

IMG_DIR = r'C:\Users\Yuval\PycharmProjects\3D_vision_final_project\calibration\ball_calibration'



# Define the return type with Mat (OpenCV) and ndarray (NumPy)
def get_images(directory: str) -> List[Tuple[Union[np.ndarray, cv2.Mat], str]]:
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
                images.append((img, filename))
    return images

def correct_and_enhance_image(image: Union[np.ndarray, cv2.Mat]) -> Union[np.ndarray, cv2.Mat]:
    # Apply CLAHE to enhance contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Remove pinhole darkness effect by applying a mask to correct the vignetting
    rows, cols = image.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, 200)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, 200)
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
    mask = mask.astype(np.uint8)
    vignette_removed = cv2.addWeighted(image, 0.8, cv2.merge((mask, mask, mask)), 0.2, 0)
    image = vignette_removed

    # Convert to HSV color space to adjust the intensity and reduce chromatic aberration
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply histogram equalization to the V channel to correct for pinhole effect and adjust brightness
    v = cv2.equalizeHist(v)

    # Merge the channels back
    v = cv2.add(v, 60)  # Increase brightness further
    corrected_hsv = cv2.merge((h, s, v))

    # Convert back to BGR color space
    corrected_image = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)
    return corrected_image


def detect_ellipses(image: Union[np.ndarray, cv2.Mat], output_dir: str, filename: str) -> list[
    dict[str, float | Sequence[float] | Any]]:

    # Neutralize chromatic colors
    image = correct_and_enhance_image(image)
    # Define the color range for a tennis ball (yellow-green)
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])

    # Create a mask for detecting the tennis ball's color
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_green = cv2.erode(mask_green, None, iterations=2)
    mask_green = cv2.dilate(mask_green, None, iterations=4)
    kernel = np.ones((10, 10), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("mask_green", mask_green)

    # Find contours in the image
    contours, _ = cv2.findContours(mask_green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []

    # Loop over the contours
    for contour in contours:
        # Only fit ellipses to contours that have more than 5 points
        if len(contour) >= 5:

            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            (center, (width, height), angle) = ellipse

            # Filter out ellipses based on size, roundness, and area (tennis ball is almost round)
            min_width, min_height = 50, 50  # Minimum size for detected tennis ball
            min_area = 1000  # Minimum area threshold to filter out small holes
            contour_area = cv2.contourArea(contour)
            aspect_ratio = width / height if width > height else height / width
            print(f"aspect ratio: {aspect_ratio}")
            if 0.7 < aspect_ratio < 1.4 and width > min_width and height > min_height and contour_area > min_area:
                ellipses.append({
                    "center": center,
                    "width": width,
                    "height": height,
                    "angle": angle
                })
                # Draw the ellipse on the image for visualization (optional)
                cv2.ellipse(image, ellipse, (0, 255, 0), 2)

                radius = get_ellipse_direction_radius(angle, center, image, width, height)
                print(f"Ball radius in direction of camera center: {radius}")

                # Add a label with the radius size
                label = f"Radius: {radius:.2f}"
                cv2.putText(image, label, (int(center[0]), int(center[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

                # Save the image with the ellipse and radius label
                output_path = os.path.join(output_dir,
                                           f"ellipse_radius_{radius:.2f}_distance_{filename.split('_')[0]}_offset_{filename.split('_')[1]}.png")
                cv2.imwrite(output_path, image)

    return ellipses, radius


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

    # resized_image = resize_image(image)
    # # Display the result
    # cv2.imshow("Radius towards Image Center", resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Return the length of the projected radius
    radius_length = np.linalg.norm(np.array(end_point) - np.array(center))
    return radius_length


def resize_image(image):
    image_copy = image.copy()
    # Resize the image by a scaling factor of 0.5 while keeping the aspect ratio consistent.
    scale_factor = 0.3
    # Calculate the new dimensions
    new_width = int(image_copy.shape[1] * scale_factor)
    new_height = int(image_copy.shape[0] * scale_factor)
    # Perform the resize operation
    resized_image = cv2.resize(image_copy, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def calibrate_camera():
    images = get_images(IMG_DIR)
    output_dir = os.path.join(IMG_DIR, "processed_images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ratios = {}
    offsets = set()
    center_radius = None
    for img, filename in images:
        ellipses, radius = detect_ellipses(img, output_dir, filename)
        for el in ellipses:
            print(f"Ellipse: Center = {el['center']}, "
                  f"Width = {el['width']}, Height = {el['height']}, "
                  f"Angle = {el['angle']}")
            distance = float(filename.split('_')[0])
            offset = int(filename.split('_')[1].split('.')[0])

            if offset == 0:
                center_radius = radius
            else:
                radius_ratio = radius / center_radius
                if offset not in ratios:
                    ratios[offset] = []
                ratios[offset].append((distance, radius_ratio))
                offsets.add(offset)

    # Prepare data for curve fitting
    if center_radius is not None:
        for offset in offsets:
            if len(ratios[offset]) < 3:
                print(f"Not enough data points for offset {offset} to perform curve fitting. Skipping...")
                continue
            distances, ratio_values = zip(*ratios[offset])

            # Define a fitting function (e.g., a quadratic curve)
            def fitting_func(x, a, b, c):
                return a * x**2 + b * x + c

            # Perform curve fitting
            params, _ = curve_fit(fitting_func, distances, ratio_values)

            # Plot the data and the fitting curve
            plt.scatter(distances, ratio_values, label=f'Data Points (Offset {offset})')
            fit_x = np.linspace(min(distances), max(distances), 100)
            fit_y = fitting_func(fit_x, *params)
            plt.plot(fit_x, fit_y, label=f'Fitting Curve (Offset {offset})')
            plt.xlabel('Distance from Camera (cm)')
            plt.ylabel('Radius Ratio')
            plt.title(f'Curve Fit of Radius Ratio vs. Distance (Offset {offset})')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'curve_fit_offset_{offset}.png'))
            plt.show()
            print(f'Fitting parameters for offset {offset}: a={params[0]}, b={params[1]}, c={params[2]}')


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