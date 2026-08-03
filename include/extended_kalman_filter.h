#ifndef EXTENDED_KALMAN_FILTER_H
#define EXTENDED_KALMAN_FILTER_H

#include <Arduino.h>
#include <math.h>
#include <cmath>

class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter(float processNoiseQ = 0.1f, 
                        float measurementNoiseR = 0.5f,
                        float initialDistance = 0.0f,
                        float initialVelocity = 0.0f);
    
    void predict(float dt);
    
    void update(float measurement, float confidence = 1.0f);
    
    float getDistance() const { return state[0]; }
    
    float getVelocity() const { return state[1]; }
    
    float getCovariance() const { return P[0][0]; }
    
    void reset(float initialDistance = 0.0f, float initialVelocity = 0.0f);
    
    bool isOutlier(float measurement, float threshold = 3.0f) const;

private:
    float state[2];
    
    float P[2][2];
    
    float Q[2][2];
    
    float R;
    
    void setTransitionMatrix(float dt);
    
    float H[2];
    
    float innovation;
    
    float S;
    
    float K[2];
    
    void matrixMultiply2x2(const float A[2][2], const float B[2][2], float C[2][2]);
    void matrixMultiply2x2x2x1(const float A[2][2], const float B[2], float C[2]);
    void matrixTranspose2x2(const float A[2][2], float AT[2][2]);
    float matrixInverse1x1(float scalar);
};

#endif
