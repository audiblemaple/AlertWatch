import pytest
import numpy as np

from . import calculate_EAR


def test_calculate_EAR_nominal():
    """
    Test the EAR calculation with a typical, symmetric set of eye landmarks.
    We'll position the eye corners horizontally and the top/bottom landmarks at known distances.
    """
    # eye landmarks (6, 2):
    #   0: left corner
    #   1: top-left
    #   2: top-right
    #   3: right corner
    #   4: bottom-right
    #   5: bottom-left
    #
    # This example:
    #   corners at (0,0) and (10,0)
    #   top points near (2,5), (8,5)
    #   bottom points near (2,1), (8,1)
    eye = np.array([
        [0, 0],   # left corner
        [2, 5],   # top-left
        [8, 5],   # top-right
        [10, 0],  # right corner
        [8, 1],   # bottom-right
        [2, 1]    # bottom-left
    ], dtype=float)

    ear = calculate_EAR(eye)

    # Let's do a manual computation for verification:
    # A = distance( top-left, bottom-left ) = distance( [2,5], [2,1] ) = 4
    # B = distance( top-right, bottom-right ) = distance( [8,5], [8,1] ) = 4
    # C = distance( left corner, right corner ) = distance( [0,0], [10,0] ) = 10
    # EAR = (A + B) / (2 * C) = (4 + 4) / (2 * 10) = 8 / 20 = 0.4
    assert pytest.approx(ear, 0.001) == 0.4


def test_calculate_EAR_zero_horizontal_distance():
    """
    If the horizontal distance between eye[0] and eye[3] is zero,
    we expect the result to be either inf or nan, because
    floating-point division by zero doesn't raise ZeroDivisionError in NumPy.
    """
    eye = np.array([
        [5, 5],  # left corner
        [5, 6],
        [5, 7],
        [5, 5],  # same as left corner
        [5, 4],
        [5, 3]
    ], dtype=float)

    ear = calculate_EAR(eye)
    # Instead of expecting an exception, we check the numeric result
    assert np.isinf(ear) or np.isnan(ear), "EAR should be inf or nan when horizontal distance is zero"



def test_calculate_EAR_identical_points():
    """
    If all points are the same, the distances are all zero.
    We expect inf or nan, not a ZeroDivisionError.
    """
    eye = np.array([
        [5, 5],
        [5, 5],
        [5, 5],
        [5, 5],
        [5, 5],
        [5, 5]
    ], dtype=float)

    ear = calculate_EAR(eye)
    assert np.isinf(ear) or np.isnan(ear), "EAR should be inf or nan when points are identical"


def test_calculate_EAR_non_symmetric_eye():
    """
    Test with a non-symmetric set of points to ensure the formula
    holds in a slightly tilted or asymmetric scenario.
    """
    # Slightly tilted eye: left corner higher, right corner lower
    eye = np.array([
        [0, 0],   # left corner
        [1, 3],   # top-left
        [4, 4],   # top-right
        [5, 2],   # right corner
        [4, 1],   # bottom-right
        [1, 2]    # bottom-left
    ], dtype=float)

    ear = calculate_EAR(eye)

    # We'll compute approximate distances:
    # A = distance( [1,3], [1,2] ) = 1
    # B = distance( [4,4], [4,1] ) = 3
    # C = distance( [0,0], [5,2] ) ~ sqrt( 5^2 + 2^2 ) = sqrt(29) ~ 5.385
    #
    # EAR ~ (1 + 3) / (2 * 5.385) = 4 / 10.77 ~ 0.371
    # We'll allow for a small margin of error
    expected = 4 / (2 * np.sqrt(29))  # ~ 0.371
    assert pytest.approx(ear, 0.001) == expected


def test_calculate_EAR_random_points():
    """
    Generate random valid eye points where the horizontal distance
    won't be zero. We just ensure the function returns a float
    without errors.
    """
    # Generate random points but ensure the horizontal corner points
    # are distinct to avoid division by zero:
    rng = np.random.default_rng(seed=42)

    # Eye corners
    left_corner = rng.random(2) * 5
    right_corner = left_corner + [5, 0]  # shift x by 5 to ensure they're different

    # Top-left / top-right / bottom-right / bottom-left
    top_left = left_corner + [1, rng.random() * 3 + 1]
    top_right = right_corner + [-1, rng.random() * 3 + 1]
    bottom_right = right_corner + [-1, -1]
    bottom_left = left_corner + [1, -1]

    eye = np.array([
        left_corner,
        top_left,
        top_right,
        right_corner,
        bottom_right,
        bottom_left
    ], dtype=float)

    ear = calculate_EAR(eye)
    # Just check it returns a float in a plausible range
    assert isinstance(ear, float)
    # Real EAR typically ranges 0.0 - 1.0 or so, but can go beyond in extreme geometry
    # We'll just check it's not infinite or NaN:
    assert not np.isnan(ear)
    assert not np.isinf(ear)
