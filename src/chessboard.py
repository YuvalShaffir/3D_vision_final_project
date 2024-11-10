import numpy as np
import cv2 as cv
import glob
import os

IMAGE_DIR = r'C:\Users\Yuval\PycharmProjects\3D_vision_final_project\calibration\Chessboard'


def calibrate_camera_sb(image_dir=IMAGE_DIR, pattern_size=(14, 9), square_size=12.1):
    """
    Calibrates the camera using images of a chessboard pattern with findChessboardCornersSB.

    Parameters:
    - image_dir: Directory containing calibration images.
    - pattern_size: Number of inner corners per chessboard row and column (rows, columns).
    - square_size: Size of a square in your defined unit (e.g., millimeters).

    Returns:
    - ret: RMS re-projection error.
    - camera_matrix: Camera matrix (intrinsic parameters).
    - dist_coeffs: Distortion coefficients.
    - rvecs: Rotation vectors.
    - tvecs: Translation vectors.
    """
    # Prepare object points based on the real-world coordinates of the chessboard corners
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Retrieve all images from the specified directory
    images = glob.glob(f'{image_dir}/*.jpg')

    for fname in images:
        print(f"filename: {fname}")
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners using the sector-based method
        ret, corners = cv.findChessboardCornersSB(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Optionally, draw and display the corners
            # cv.drawChessboardCorners(img, pattern_size, corners, ret)
            # cv.imshow('Chessboard Corners', img)
            # cv.waitKey(500)

    cv.destroyAllWindows()

    # Perform camera calibration to obtain the camera matrix and distortion coefficients
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


# Example usage:
# print(calibrate_camera_sb(r'C:\Users\Yuval\PycharmProjects\3D_vision_final_project\calibration\rounded_chessboard'))
