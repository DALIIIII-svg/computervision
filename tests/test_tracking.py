import numpy as np

from kombat_vision.vision.tracking import KalmanFilter2D


def test_kalman_estimates_constant_velocity():
    kf = KalmanFilter2D(process_noise=0.5, measurement_noise=2.0)
    rng = np.random.default_rng(0)
    for t in range(60):
        kf.predict()
        kf.update(10 + 3.0 * t + rng.normal(0, 1), 50 - 2.0 * t + rng.normal(0, 1))
    vx, vy = kf.velocity
    assert abs(vx - 3.0) < 0.5 and abs(vy + 2.0) < 0.5


def test_kalman_predicts_when_measurement_missing():
    kf = KalmanFilter2D()
    for t in range(30):
        kf.predict()
        kf.update(5.0 * t, 0.0)
    x_before = kf.position[0]
    kf.predict()
    assert kf.position[0] > x_before + 3   # keeps moving without a measurement
