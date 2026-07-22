#include "extended_kalman_filter.h"

ExtendedKalmanFilter::ExtendedKalmanFilter(float processNoiseQ, 
                                         float measurementNoiseR,
                                         float initialDistance,
                                         float initialVelocity) {
    // Initialize state vector [distance, velocity]
    state[0] = initialDistance;
    state[1] = initialVelocity;
    
    // Initialize state covariance matrix (initial uncertainty)
    P[0][0] = 1.0f;  // Distance uncertainty
    P[0][1] = 0.0f;
    P[1][0] = 0.0f;
    P[1][1] = 1.0f;  // Velocity uncertainty
    
    // Initialize process noise covariance matrix
    // Q represents uncertainty in the motion model
    // Higher values allow the filter to adapt more quickly to changes
    Q[0][0] = processNoiseQ;
    Q[0][1] = 0.0f;
    Q[1][0] = 0.0f;
    Q[1][1] = processNoiseQ * 0.1f;  // Less uncertainty in velocity changes
    
    // Initialize measurement noise
    R = measurementNoiseR;
    
    // Initialize measurement Jacobian
    H[0] = 1.0f;  // ∂h/∂distance = 1 (we measure distance directly)
    H[1] = 0.0f;  // ∂h/∂velocity = 0 (velocity not directly measured)
}

void ExtendedKalmanFilter::predict(float dt) {
    // State transition matrix F for constant velocity model:
    // F = [1  dt]
    //     [0   1]
    // This models: distance_new = distance_old + velocity * dt
    //              velocity_new = velocity_old
    
    // Predict state: x_pred = F * x_prev
    float distance_pred = state[0] + state[1] * dt;
    float velocity_pred = state[1];
    
    state[0] = distance_pred;
    state[1] = velocity_pred;
    
    // Predict covariance: P_pred = F * P_prev * F^T + Q
    // Manual matrix multiplication to avoid dynamic allocation
    
    // F * P
    float FP[2][2];
    FP[0][0] = 1.0f * P[0][0] + dt * P[1][0];
    FP[0][1] = 1.0f * P[0][1] + dt * P[1][1];
    FP[1][0] = 0.0f * P[0][0] + 1.0f * P[1][0];
    FP[1][1] = 0.0f * P[0][1] + 1.0f * P[1][1];
    
    // F * P * F^T
    float FPT[2][2];
    FPT[0][0] = FP[0][0] * 1.0f + FP[0][1] * 0.0f;
    FPT[0][1] = FP[0][0] * dt + FP[0][1] * 1.0f;
    FPT[1][0] = FP[1][0] * 1.0f + FP[1][1] * 0.0f;
    FPT[1][1] = FP[1][0] * dt + FP[1][1] * 1.0f;
    
    // Add process noise: P_pred = FPT + Q
    P[0][0] = FPT[0][0] + Q[0][0];
    P[0][1] = FPT[0][1] + Q[0][1];
    P[1][0] = FPT[1][0] + Q[1][0];
    P[1][1] = FPT[1][1] + Q[1][1];
}

void ExtendedKalmanFilter::update(float measurement, float confidence) {
    // Adapt measurement noise based on confidence
    // Lower confidence = higher measurement noise
    float adaptedR = R / (confidence * confidence + 0.01f);
    
    // Innovation: y = z - h(x_pred)
    // For direct distance measurement: h(x) = distance
    innovation = measurement - state[0];
    
    // Innovation covariance: S = H * P * H^T + R
    // H = [1 0], so H * P * H^T = P[0][0]
    S = P[0][0] + adaptedR;
    
    // Kalman gain: K = P * H^T * S^-1
    // P * H^T = [P[0][0], P[1][0]]^T
    K[0] = P[0][0] / S;
    K[1] = P[1][0] / S;
    
    // Update state: x = x_pred + K * innovation
    state[0] = state[0] + K[0] * innovation;
    state[1] = state[1] + K[1] * innovation;
    
    // Update covariance: P = (I - K * H) * P
    // I - K * H = [1-K[0]  0     ]
    //             [ -K[1]  1     ]
    float KH[2][2];
    KH[0][0] = K[0] * H[0];
    KH[0][1] = K[0] * H[1];
    KH[1][0] = K[1] * H[0];
    KH[1][1] = K[1] * H[1];
    
    float I_KH[2][2];
    I_KH[0][0] = 1.0f - KH[0][0];
    I_KH[0][1] = 0.0f - KH[0][1];
    I_KH[1][0] = 0.0f - KH[1][0];
    I_KH[1][1] = 1.0f - KH[1][1];
    
    // P_new = (I - K*H) * P
    float P_new[2][2];
    P_new[0][0] = I_KH[0][0] * P[0][0] + I_KH[0][1] * P[1][0];
    P_new[0][1] = I_KH[0][0] * P[0][1] + I_KH[0][1] * P[1][1];
    P_new[1][0] = I_KH[1][0] * P[0][0] + I_KH[1][1] * P[1][0];
    P_new[1][1] = I_KH[1][0] * P[0][1] + I_KH[1][1] * P[1][1];
    
    // Ensure symmetry and positive definiteness
    P[0][0] = P_new[0][0];
    P[0][1] = P_new[0][1];
    P[1][0] = P_new[1][0];
    P[1][1] = P_new[1][1];
    
    // Symmetrize covariance matrix
    P[0][1] = (P[0][1] + P[1][0]) * 0.5f;
    P[1][0] = P[0][1];
}

void ExtendedKalmanFilter::reset(float initialDistance, float initialVelocity) {
    state[0] = initialDistance;
    state[1] = initialVelocity;
    
    P[0][0] = 1.0f;
    P[0][1] = 0.0f;
    P[1][0] = 0.0f;
    P[1][1] = 1.0f;
}

bool ExtendedKalmanFilter::isOutlier(float measurement, float threshold) const {
    // Mahalanobis distance: sqrt((z - x)^T * S^-1 * (z - x))
    // For scalar measurement: |z - x| / sqrt(S)
    float residual = measurement - state[0];
    float mahalanobis = fabs(residual) / sqrt(P[0][0] + R);
    
    return mahalanobis > threshold;
}

// Helper functions for matrix operations (avoid dynamic allocation)
void ExtendedKalmanFilter::matrixMultiply2x2(const float A[2][2], const float B[2][2], float C[2][2]) {
    C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0];
    C[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1];
    C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0];
    C[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1];
}

void ExtendedKalmanFilter::matrixMultiply2x2x2x1(const float A[2][2], const float B[2], float C[2]) {
    C[0] = A[0][0] * B[0] + A[0][1] * B[1];
    C[1] = A[1][0] * B[0] + A[1][1] * B[1];
}

void ExtendedKalmanFilter::matrixTranspose2x2(const float A[2][2], float AT[2][2]) {
    AT[0][0] = A[0][0];
    AT[0][1] = A[1][0];
    AT[1][0] = A[0][1];
    AT[1][1] = A[1][1];
}

float ExtendedKalmanFilter::matrixInverse1x1(float scalar) {
    return 1.0f / scalar;
}
