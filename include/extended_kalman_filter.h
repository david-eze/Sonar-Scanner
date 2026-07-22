#ifndef EXTENDED_KALMAN_FILTER_H
#define EXTENDED_KALMAN_FILTER_H

#include <Arduino.h>
#include <math.h>
#include <cmath>

/**
 * Extended Kalman Filter (EKF) for Sonar Sensor Noise Filtering
 * 
 * This EKF implementation filters multipath noise, sensor jitter, and false returns
 * from ultrasonic sensor measurements. It models the system state as [distance, velocity]
 * and uses non-linear measurement equations to handle the physics of ultrasonic ranging.
 * 
 * Mathematical Model:
 * - State vector: x = [distance, velocity]^T
 * - Process model: x_k = F * x_{k-1} + w_k (linear constant velocity model)
 * - Measurement model: z_k = h(x_k) + v_k (non-linear distance measurement)
 * 
 * The EKF linearizes the non-linear measurement function around the current state estimate
 * using the Jacobian matrix H = ∂h/∂x evaluated at the predicted state.
 */
class ExtendedKalmanFilter {
public:
    /**
     * Constructor - Initialize EKF with default parameters
     * 
     * @param processNoiseQ Process noise covariance (uncertainty in motion model)
     * @param measurementNoiseR Measurement noise covariance (sensor uncertainty)
     * @param initialDistance Initial distance estimate (meters)
     * @param initialVelocity Initial velocity estimate (m/s)
     */
    ExtendedKalmanFilter(float processNoiseQ = 0.1f, 
                        float measurementNoiseR = 0.5f,
                        float initialDistance = 0.0f,
                        float initialVelocity = 0.0f);
    
    /**
     * Predict step - Propagate state estimate using motion model
     * 
     * This implements the time update equation:
     * x_pred = F * x_prev
     * P_pred = F * P_prev * F^T + Q
     * 
     * @param dt Time step since last update (seconds)
     */
    void predict(float dt);
    
    /**
     * Update step - Incorporate new measurement
     * 
     * This implements the measurement update equation:
     * K = P_pred * H^T * (H * P_pred * H^T + R)^-1
     * x = x_pred + K * (z - h(x_pred))
     * P = (I - K * H) * P_pred
     * 
     * @param measurement Raw distance measurement (meters)
     * @param confidence Measurement confidence (0.0 to 1.0, used to adapt R)
     */
    void update(float measurement, float confidence = 1.0f);
    
    /**
     * Get current filtered distance estimate
     * @return Estimated distance (meters)
     */
    float getDistance() const { return state[0]; }
    
    /**
     * Get current velocity estimate
     * @return Estimated velocity (m/s)
     */
    float getVelocity() const { return state[1]; }
    
    /**
     * Get current estimate covariance (uncertainty)
     * @return Covariance matrix diagonal element for distance
     */
    float getCovariance() const { return P[0][0]; }
    
    /**
     * Reset filter to initial conditions
     * @param initialDistance Initial distance estimate
     * @param initialVelocity Initial velocity estimate
     */
    void reset(float initialDistance = 0.0f, float initialVelocity = 0.0f);
    
    /**
     * Check if measurement is an outlier using Mahalanobis distance
     * @param measurement Raw measurement to check
     * @param threshold Threshold for outlier detection (typically 3-5 sigma)
     * @return true if measurement is likely an outlier
     */
    bool isOutlier(float measurement, float threshold = 3.0f) const;

private:
    // State vector: [distance, velocity]
    float state[2];
    
    // State covariance matrix (2x2)
    float P[2][2];
    
    // Process noise covariance (2x2)
    float Q[2][2];
    
    // Measurement noise covariance (scalar)
    float R;
    
    // State transition matrix (2x2) - constant velocity model
    // F = [1 dt]
    //     [0  1]
    void setTransitionMatrix(float dt);
    
    // Measurement Jacobian matrix (1x2)
    // H = [1 0] for direct distance measurement
    float H[2];
    
    // Innovation (measurement residual)
    float innovation;
    
    // Innovation covariance
    float S;
    
    // Kalman gain (2x1)
    float K[2];
    
    // Helper function for matrix operations (avoid dynamic allocation)
    void matrixMultiply2x2(const float A[2][2], const float B[2][2], float C[2][2]);
    void matrixMultiply2x2x2x1(const float A[2][2], const float B[2], float C[2]);
    void matrixTranspose2x2(const float A[2][2], float AT[2][2]);
    float matrixInverse1x1(float scalar);
};

#endif // EXTENDED_KALMAN_FILTER_H
