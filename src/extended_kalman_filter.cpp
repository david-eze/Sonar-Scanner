#include "extended_kalman_filter.h"

ExtendedKalmanFilter::ExtendedKalmanFilter(float processNoiseQ, 
                                           float measurementNoiseR,
                                           float initialDistance,
                                           float initialVelocity) {
    state[0] = initialDistance;
    state[1] = initialVelocity;
    
    P[0][0] = 1.0f;
    P[0][1] = 0.0f;
    P[1][0] = 0.0f;
    P[1][1] = 1.0f;
    
    Q[0][0] = processNoiseQ;
    Q[0][1] = 0.0f;
    Q[1][0] = 0.0f;
    Q[1][1] = processNoiseQ * 0.1f;
    
    R = measurementNoiseR;
    
    H[0] = 1.0f;
    H[1] = 0.0f;
}

void ExtendedKalmanFilter::predict(float dt) {
    float distance_pred = state[0] + state[1] * dt;
    float velocity_pred = state[1];
    
    state[0] = distance_pred;
    state[1] = velocity_pred;
    
    float FP[2][2];
    FP[0][0] = 1.0f * P[0][0] + dt * P[1][0];
    FP[0][1] = 1.0f * P[0][1] + dt * P[1][1];
    FP[1][0] = 0.0f * P[0][0] + 1.0f * P[1][0];
    FP[1][1] = 0.0f * P[0][1] + 1.0f * P[1][1];
    
    float FPT[2][2];
    FPT[0][0] = FP[0][0] * 1.0f + FP[0][1] * 0.0f;
    FPT[0][1] = FP[0][0] * dt + FP[0][1] * 1.0f;
    FPT[1][0] = FP[1][0] * 1.0f + FP[1][1] * 0.0f;
    FPT[1][1] = FP[1][0] * dt + FP[1][1] * 1.0f;
    
    P[0][0] = FPT[0][0] + Q[0][0];
    P[0][1] = FPT[0][1] + Q[0][1];
    P[1][0] = FPT[1][0] + Q[1][0];
    P[1][1] = FPT[1][1] + Q[1][1];
}

void ExtendedKalmanFilter::update(float measurement, float confidence) {
    float adaptedR = R / (confidence * confidence + 0.01f);
    
    innovation = measurement - state[0];
    
    S = P[0][0] + adaptedR;
    
    K[0] = P[0][0] / S;
    K[1] = P[1][0] / S;
    
    state[0] = state[0] + K[0] * innovation;
    state[1] = state[1] + K[1] * innovation;
    
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
    
    float P_new[2][2];
    P_new[0][0] = I_KH[0][0] * P[0][0] + I_KH[0][1] * P[1][0];
    P_new[0][1] = I_KH[0][0] * P[0][1] + I_KH[0][1] * P[1][1];
    P_new[1][0] = I_KH[1][0] * P[0][0] + I_KH[1][1] * P[1][0];
    P_new[1][1] = I_KH[1][0] * P[0][1] + I_KH[1][1] * P[1][1];
    
    P[0][0] = P_new[0][0];
    P[0][1] = P_new[0][1];
    P[1][0] = P_new[1][0];
    P[1][1] = P_new[1][1];
    
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
    float residual = measurement - state[0];
    float mahalanobis = fabs(residual) / sqrt(P[0][0] + R);
    
    return mahalanobis > threshold;
}

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
