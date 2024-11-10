import os
import cv2
import numpy as np
from typing import List, Union, Dict, Any, Sequence, Tuple
import chessboard as chess
import LinearModel

IMG_DIR = r'C:\Users\Yuval\PycharmProjects\3D_vision_final_project\calibration\new_ball_calibration'
TEST_DIR = r'C:\Users\Yuval\PycharmProjects\3D_vision_final_project\test_images'


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

    # # Remove pinhole darkness effect by applying a mask to correct the vignetting
    # rows, cols = image.shape[:2]
    # X_resultant_kernel = cv2.getGaussianKernel(cols, 200)
    # Y_resultant_kernel = cv2.getGaussianKernel(rows, 200)
    # resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
    # mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
    # mask = mask.astype(np.uint8)
    # vignette_removed = cv2.addWeighted(image, 0.8, cv2.merge((mask, mask, mask)), 0.2, 0)
    # image = vignette_removed

    # Convert to HSV color space to adjust the intensity and reduce chromatic aberration
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply histogram equalization to the V channel to correct for pinhole effect and adjust brightness
    v = cv2.equalizeHist(v)

    # Merge the channels back
    v = cv2.add(v, 10)  # Increase brightness further
    corrected_hsv = cv2.merge((h, s, v))

    # Convert back to BGR color space
    corrected_image = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)
    return corrected_image


def detect_ellipses(image: Union[np.ndarray, cv2.Mat], output_dir: str, filename: str, new_camera_matrix) -> list[
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
    mask_green = cv2.erode(mask_green, None, iterations=1)
    mask_green = cv2.dilate(mask_green, None, iterations=2)
    kernel = np.ones((5, 5), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    # mask_green2 = resize_image(mask_green)
    # cv2.imshow("mask_green", mask_green2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find contours in the image
    contours, _ = cv2.findContours(mask_green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    radius = None
    # Loop over the contours
    for contour in contours:
        # Only fit ellipses to contours that have more than 5 points
        if len(contour) >= 5:

            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            (center, (width, height), angle) = ellipse

            # Filter out ellipses based on size, roundness, and area (tennis ball is almost round)
            min_width, min_height = 10, 10  # Minimum size for detected tennis ball
            min_area = 2000  # Minimum area threshold to filter out small holes
            contour_area = cv2.contourArea(contour)
            aspect_ratio = width / height if width > height else height / width
            if 0.7 < aspect_ratio < 1.2 and width > min_width and height > min_height and contour_area > min_area:

                print(f"aspect ratio: {aspect_ratio}\n area: {contour_area}")

                ellipses.append({
                    "center": center,
                    "width": width,
                    "height": height,
                    "angle": angle
                })
                # Draw the ellipse on the image for visualization (optional)
                cv2.ellipse(image, ellipse, (0, 255, 0), 2)

                radius = get_ellipse_direction_radius(angle, center, image, width, height, new_camera_matrix)
                print(f"Ball radius in direction of camera center: {radius}")

                # Add a label with the radius size
                label = f"Radius: {radius:.2f}"
                cv2.putText(image, label, (int(center[0]), int(center[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
                base_name = os.path.splitext(os.path.basename(filename))[0]
                # Save the image with the ellipse and radius label
                if base_name.split('_') == 2:
                    output_path = os.path.join(output_dir,
                                               f"ellipse_radius_{radius:.2f}_distance_{base_name.split('_')[0]}_offset_{base_name.split('_')[1]}.png")
                else:
                    output_path = os.path.join(output_dir, f"{base_name}.png")
                cv2.imwrite(output_path, image)

    if not radius:
        return ellipses, None

    return ellipses, radius


def get_ellipse_direction_radius(angle, center, image, width, height, new_camera_matrix):
    # Calculate the direction vector from image center to ellipse center
    image_center = get_image_center(new_camera_matrix)
    direction_vector = (image_center[0] - center[0], image_center[1] - center[1])

    # Convert angle to radians for trigonometric functions
    angle_rad = np.deg2rad(angle)

    # Major axis vector (based on the longer dimension of the ellipse)
    major_axis_vector = (np.cos(angle_rad) * width / 2, np.sin(angle_rad) * width / 2)

    # Minor axis vector (based on the shorter dimension of the ellipse)
    minor_axis_vector = (-np.sin(angle_rad) * height / 2, np.cos(angle_rad) * height / 2)

    # Normalize the direction vector
    direction_magnitude = np.linalg.norm(direction_vector)
    if direction_magnitude == 0:
        return 0  # If the direction vector has zero magnitude, return zero distance
    normalized_direction_vector = (direction_vector[0] / direction_magnitude, direction_vector[1] / direction_magnitude)

    # Project the normalized direction vector onto the major and minor axes
    projection_major = np.dot(normalized_direction_vector, major_axis_vector) / np.linalg.norm(major_axis_vector)
    projection_minor = np.dot(normalized_direction_vector, minor_axis_vector) / np.linalg.norm(minor_axis_vector)

    # Calculate the final endpoint along the major or minor axis based on the larger projection
    if abs(projection_major) > abs(projection_minor):
        end_point = (
            center[0] + np.sign(projection_major) * major_axis_vector[0],
            center[1] + np.sign(projection_major) * major_axis_vector[1]
        )
    else:
        end_point = (
            center[0] + np.sign(projection_minor) * minor_axis_vector[0],
            center[1] + np.sign(projection_minor) * minor_axis_vector[1]
        )

    # Ensure the line is pointing towards the center of the image
    end_vector = (end_point[0] - center[0], end_point[1] - center[1])
    dot_product = end_vector[0] * direction_vector[0] + end_vector[1] * direction_vector[1]

    if dot_product < 0:
        # Flip the endpoint to point toward the image center
        end_point = (
            center[0] - end_vector[0],
            center[1] - end_vector[1]
        )

    # Draw the line from the ellipse center to the endpoint
    cv2.line(image, (int(center[0]), int(center[1])), (int(end_point[0]), int(end_point[1])), (255, 0, 0), 2)

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


# Create a directory to save the undistorted images
calibrated_dir = os.path.join(IMG_DIR, 'calibrated_images')
os.makedirs(calibrated_dir, exist_ok=True)


def undistort_image(img, filename, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop the image based on the region of interest
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y + h, x:x + w]

    # Save the undistorted image

    base_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(calibrated_dir, base_name + '.png')
    cv2.imwrite(save_path, undistorted_img)
    return undistorted_img, new_camera_matrix


def calculate_distance(ball_center, image_center):
    # Unpack the coordinates for clarity
    x1, y1 = ball_center
    x2, y2 = image_center

    # Calculate Euclidean distance
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def get_image_center(new_camera_matrix):
    # The principal point (cx', cy') in the new camera matrix represents the new image center
    cx = new_camera_matrix[0, 2]
    cy = new_camera_matrix[1, 2]

    return (int(cx), int(cy))


def calibrate_camera():
    images = get_images(IMG_DIR)
    output_dir = os.path.join(IMG_DIR, "processed_images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ret, camera_matrix, dist_coeffs, rvecs, tvecs = chess.calibrate_camera_sb()
    # print(f'ret: {ret}\n camera_matrix: {camera_matrix}\n dist_coeffs: {dist_coeffs}\n rvecs: {rvecs}\n tvecs: {rvecs}\n')

    camera_matrix = np.array([[2.89806869e+03, 0.00000000e+00, 1.53624044e+03],
                              [0.00000000e+00, 2.89816682e+03, 2.05761760e+03],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    dist_coeffs = np.array([[2.05489723e-01, -6.73059010e-01, 3.36320714e-04, 1.49915927e-04, 6.66189722e-01]])

    ratios = {}
    offsets = set()
    center_radius = None
    for img, filename in images:
        undistorted_img, new_camera_matrix = undistort_image(img, filename, camera_matrix, dist_coeffs)
        ellipses, radius = detect_ellipses(undistorted_img, output_dir, filename, new_camera_matrix)
        if not radius:
            continue

        for el in ellipses:
            print(f"Ellipse: Center = {el['center']}, "
                  f"Width = {el['width']}, Height = {el['height']}, "
                  f"Angle = {el['angle']}")
            # camera_height = float(filename.split('_')[0])
            base_name = os.path.splitext(os.path.basename(filename))[0]
            offset = int(base_name.split('_')[1])
            distance_from_center = calculate_distance(el['center'], get_image_center(new_camera_matrix))

            if offset == 0:
                center_radius = radius
            else:
                radius_ratio = radius / center_radius
                if offset not in ratios:
                    ratios[offset] = []
                # ratios[offset].append((distance, radius_ratio))
                ratios[offset].append([distance_from_center, radius, center_radius, el['width'], el['height'], el['angle']])
                offsets.add(offset)

    # # Prepare data for curve fitting
    # if center_radius is not None:
    #     for offset in offsets:
    #         # if len(ratios[offset]) < 3:
    #         #     print(f"Not enough data points for offset {offset} to perform curve fitting. Skipping...")
    #         #     continue
    #         distances, radius, center_radius = zip(*ratios[offset])
    #         distances = np.array(distances)  # Convert to NumPy array
    #         radius = np.array(radius)  # Convert to NumPy array
    #         center_radius = np.array(center_radius)  # Convert to NumPy array
    #
    #         # Element-wise division to get the ratio values
    #         ratio_values = radius / center_radius
    #         # Define a fitting function (e.g., a quadratic curve)
    #         def fitting_func(x, a, b, c):
    #             return a * x ** 2 + b * x + c
    #
    #         # Perform curve fitting
    #         # params, _ = curve_fit(fitting_func, distances, ratio_values)
    #
    #         # Plot the data and the fitting curve
    #         plt.scatter(distances, ratio_values, label=f'Data Points (Offset {offset})')
    #         # fit_x = np.linspace(min(distances), max(distances), 100)
    #         # fit_y = fitting_func(fit_x, *params)
    #         # plt.plot(fit_x, fit_y, label=f'Fitting Curve (Offset {offset})')
    #         plt.xlabel('Distance from Camera (cm)')
    #         plt.ylabel('Radius Ratio')
    #         plt.title(f'Curve Fit of Radius Ratio vs. Distance (Offset {offset})')
    #         plt.legend()
    #         plt.savefig(os.path.join(output_dir, f'curve_fit_offset_{offset}.png'))
    #         plt.show()
    #         # print(f'Fitting parameters for offset {offset}: a={params[0]}, b={params[1]}, c={params[2]}')

    # Prepare data for curve fitting
    if center_radius is not None:
        plt.figure(figsize=(10, 6))  # Create a single figure for all data points
        for offset in offsets:
            # if len(ratios[offset]) < 3:
            #     print(f"Not enough data points for offset {offset} to perform curve fitting. Skipping...")
            #     continue
            distances, radius, center_radius, width, height, angle = zip(*ratios[offset])
            distances = np.array(distances)  # Convert to NumPy array
            radius = np.array(radius)  # Convert to NumPy array
            center_radius = np.array(center_radius)  # Convert to NumPy array

            # Element-wise division to get the ratio values
            ratio_values = radius / center_radius

            # Plot the data points for each offset on the same figure
            plt.scatter(distances, ratio_values, label=f'Data Points (Offset {offset})')

        # Set plot labels and title
        plt.xlabel('Distance from Camera (cm)')
        plt.ylabel('Radius Ratio')
        plt.title('Curve Fit of Radius Ratio vs. Distance for All Offsets')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'curve_fit_all_offsets.png'))
        plt.show()

    model = LinearModel.create_prediction_model(ratios)
    return model


def predict_ball_distance(directory, model):
    images = get_images(directory)
    output_dir = os.path.join(directory, "processed_images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ret, camera_matrix, dist_coeffs, rvecs, tvecs = chess.calibrate_camera_sb()
    # print(f'ret: {ret}\n camera_matrix: {camera_matrix}\n dist_coeffs: {dist_coeffs}\n rvecs: {rvecs}\n tvecs: {rvecs}\n')
    camera_matrix = np.array([[2.89806869e+03, 0.00000000e+00, 1.53624044e+03],
                             [0.00000000e+00, 2.89816682e+03, 2.05761760e+03],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    dist_coeffs = np.array([[2.05489723e-01, -6.73059010e-01, 3.36320714e-04, 1.49915927e-04, 6.66189722e-01]])


    for img, filename in images:
        undistorted_img, new_camera_matrix = undistort_image(img, filename, camera_matrix, dist_coeffs)
        ellipses, radius = detect_ellipses(undistorted_img, output_dir, filename, new_camera_matrix)
        if not radius:
            continue

        for el in ellipses:
            print(f"Ellipse: Center = {el['center']}, "
                  f"Width = {el['width']}, Height = {el['height']}, "
                  f"Angle = {el['angle']}")
            distance_from_center = calculate_distance(el['center'], get_image_center(new_camera_matrix))
            aspect_ratio = el['width'] / el['height']
            ellipse_area = np.pi * (el['width'] / 2) * (el['height'] / 2)

            predicted_center_radius = LinearModel.predict_center_radius(model, distance_from_center, radius,
                                                                        aspect_ratio, ellipse_area, el['angle'])
            print(f"predicted center radius: {predicted_center_radius}")

            # Extract the focal length from the camera matrix (typically fx or fy) in pixels
            fx = new_camera_matrix[0, 0]
            fy = new_camera_matrix[1, 1]
            focal_length = (fx + fy) // 2
            # samsung A22 camera has 4.7 mm focal length
            # focal_length = 4.7
            # Known real-world size of the object (e.g., the diameter of the ball in cm)
            real_world_size = 6.4

            # Apparent size of the object in the image (e.g., the diameter in pixels)
            object_size_in_image = predicted_center_radius * 2

            distance_to_camera = calculate_distance_to_camera(focal_length, real_world_size, object_size_in_image)
            print(f"Distance to the object from the camera: {distance_to_camera:.2f} cm")
            print(f"Real distance is: {os.path.splitext(os.path.basename(filename))[0]}")


def calculate_distance_to_camera(focal_length, real_world_size, object_size_in_image):
    """
    Calculate the distance from the camera to the object.

    :param focal_length: Focal length of the camera (in pixels).
    :param real_world_size: Real-world size of the object (e.g., diameter in cm).
    :param object_size_in_image: Apparent size of the object in the image (in pixels).
    :return: Distance to the object (in the same units as real_world_size).
    """
    if object_size_in_image == 0:
        raise ValueError("Object size in the image must be greater than zero to avoid division by zero.")

    distance = (focal_length * real_world_size) / object_size_in_image
    return distance


if __name__ == '__main__':
    model = calibrate_camera()
    predict_ball_distance(TEST_DIR, model)
