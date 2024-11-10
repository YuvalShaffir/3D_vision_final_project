# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
#
#
# def create_prediction_model(ratios):
#     data_points = []
#     for offset, data in ratios.items():
#         for entry in data:
#             distance_from_center = entry[0]  # Distance from the center of the camera (pixels)
#             radius = entry[1]  # Measured radius at the given location
#             center_radius = entry[2]  # Center radius on the same plane (target)
#
#             # Append center coordinates (x, y), radius, and distance as features, and center_radius as the target
#             data_points.append([distance_from_center, radius, center_radius])
#
#     # Convert to NumPy arrays for fitting
#     data_points = np.array(data_points)
#     X = data_points[:, :2]  # Features: distance_from_center, radius
#     y = data_points[:, 2]  # Target: center_radius
#
#     # Train a linear regression model
#     model = LinearRegression()
#     model.fit(X, y)
#
#     return model
#
#
# def predict_center_radius(model, distance_from_center, input_radius):
#     # Prepare the input data for prediction
#     input_features = np.array([[distance_from_center, input_radius]])
#     predicted_center_radius = model.predict(input_features)
#     return predicted_center_radius[0]

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline


def create_prediction_model(ratios):
    data_points = []
    for offset, data in ratios.items():
        for entry in data:
            distance_from_center = entry[0]  # Distance from the center of the camera (pixels)
            radius = entry[1]  # Measured radius at the given location
            center_radius = entry[2]  # Center radius on the same plane (target)
            width = entry[3]  # Width of the ellipse
            height = entry[4]  # Height of the ellipse
            angle = entry[5]  # Angle of the ellipse

            # Calculate additional features
            aspect_ratio = width / height
            ellipse_area = np.pi * (width / 2) * (height / 2)

            ratio = center_radius / radius
            # Append features for training: distance_from_center, radius, aspect_ratio, ellipse_area, angle
            data_points.append([distance_from_center, radius, aspect_ratio, ellipse_area, angle, ratio])

    # Convert to NumPy arrays for fitting
    data_points = np.array(data_points)
    X = data_points[:, :5]  # Features: distance_from_center, radius, aspect_ratio, ellipse_area, angle
    y = data_points[:, 5]  # Target: center_radius

    # Train an SVR model with an RBF kernel
    model = make_pipeline(PolynomialFeatures(degree=4), SVR(kernel='rbf', C=10, epsilon=0.1))
    model.fit(X, y)

    return model


def predict_center_radius(model, distance_from_center, input_radius, aspect_ratio, ellipse_area, angle):
    # Prepare the input data for prediction
    input_features = np.array([[distance_from_center, input_radius, aspect_ratio, ellipse_area, angle]])
    predicted_ratio = model.predict(input_features)
    center_radius = predicted_ratio[0] * input_radius
    return center_radius
