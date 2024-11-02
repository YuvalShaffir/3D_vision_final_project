import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def create_prediction_model(ratios):
    data_points = []
    for offset, data in ratios.items():
        for entry in data:
            distance_from_center = entry[0]  # Distance from the center of the camera (pixels)
            radius = entry[1]  # Measured radius at the given location
            center_radius = entry[2]  # Center radius on the same plane (target)

            # Append center coordinates (x, y), radius, and distance as features, and center_radius as the target
            data_points.append([distance_from_center, radius, center_radius])

    # Convert to NumPy arrays for fitting
    data_points = np.array(data_points)
    X = data_points[:, :2]  # Features: distance_from_center, radius
    y = data_points[:, 2]  # Target: center_radius

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    return model


def predict_center_radius(model, distance_from_center, input_radius):
    # Prepare the input data for prediction
    input_features = np.array([[distance_from_center, input_radius]])
    predicted_center_radius = model.predict(input_features)
    return predicted_center_radius[0]