#ifndef OBJECT_CLASSIFIER_H
#define OBJECT_CLASSIFIER_H

#include <Arduino.h>
#include <math.h>
#include <cmath>
#include <limits>

/**
 * Lightweight ML Classifier for Sonar Object Classification
 * 
 * This classifier analyzes consecutive sonar echo signatures to categorize targets
 * into distinct object classes using a k-Nearest Neighbors (k-NN) approach.
 * 
 * Feature Extraction Pipeline:
 * 1. Distance change velocity (radial velocity of target)
 * 2. Return variance (consistency of measurements)
 * 3. Surface reflection profile (echo amplitude characteristics)
 * 4. Temporal correlation (how measurements change over time)
 * 
 * Object Classes:
 * - WALL_FLAT: Flat surfaces with consistent reflections
 * - CORNER_EDGE: Sharp discontinuities with high variance
 * - DYNAMIC_MOVING: Objects with significant velocity
 * - HUMAN_SOFT: Absorbing materials with weak returns
 * - UNKNOWN: Insufficient data for classification
 */
class ObjectClassifier {
public:
    enum ObjectClass {
        UNKNOWN = 0,
        WALL_FLAT,
        CORNER_EDGE,
        DYNAMIC_MOVING,
        HUMAN_SOFT,
        NUM_CLASSES
    };
    
    // Feature vector structure (fixed-size to avoid dynamic allocation)
    struct FeatureVector {
        float distance;           // Current distance (meters)
        float velocity;           // Radial velocity (m/s)
        float variance;           // Measurement variance
        float amplitude;          // Echo amplitude/confidence
        float gradient;           // Distance change rate
        float consistency;        // Measurement consistency (0-1)
    };
    
    // Training sample structure
    struct TrainingSample {
        FeatureVector features;
        ObjectClass label;
    };
    
    /**
     * Constructor - Initialize classifier with training data
     * @param k Number of neighbors for k-NN (typically 3-5)
     */
    ObjectClassifier(uint8_t k = 3);
    
    /**
     * Extract features from recent measurement history
     * @param distances Array of recent distance measurements
     * @param confidences Array of measurement confidences
     * @param count Number of samples in arrays
     * @param dt Time between measurements (seconds)
     * @return Extracted feature vector
     */
    FeatureVector extractFeatures(const float* distances, 
                                 const float* confidences,
                                 uint8_t count,
                                 float dt);
    
    /**
     * Classify object based on feature vector
     * @param features Feature vector to classify
     * @return Predicted object class
     */
    ObjectClass classify(const FeatureVector& features);
    
    /**
     * Get class name as string
     * @param cls Object class
     * @return Class name string
     */
    static const char* getClassName(ObjectClass cls);
    
    /**
     * Get classification confidence (0.0 to 1.0)
     * @return Confidence of last classification
     */
    float getConfidence() const { return lastConfidence; }
    
    /**
     * Initialize training data with pre-defined samples
     * This loads a basic decision tree/k-NN model
     */
    void initializeTrainingData();
    
    /**
     * Add custom training sample
     * @param sample Training sample to add
     */
    void addTrainingSample(const TrainingSample& sample);
    
    /**
     * Reset classifier state
     */
    void reset();

private:
    static const uint8_t MAX_TRAINING_SAMPLES = 50;
    static const uint8_t MAX_HISTORY = 10;
    
    uint8_t kNeighbors;
    TrainingSample trainingData[MAX_TRAINING_SAMPLES];
    uint8_t numTrainingSamples;
    float lastConfidence;
    
    // Measurement history for feature extraction
    float historyDistances[MAX_HISTORY];
    float historyConfidences[MAX_HISTORY];
    uint8_t historyIndex;
    uint8_t historyCount;
    
    /**
     * Calculate Euclidean distance between feature vectors
     * @param a First feature vector
     * @param b Second feature vector
     * @return Euclidean distance
     */
    float calculateDistance(const FeatureVector& a, const FeatureVector& b);
    
    /**
     * Normalize feature vector to [0, 1] range
     * @param features Feature vector to normalize
     */
    void normalizeFeatures(FeatureVector& features);
    
    /**
     * Feature normalization parameters (min/max values)
     */
    struct FeatureBounds {
        float min;
        float max;
    };
    FeatureBounds featureBounds[6];
    
    /**
     * Find k nearest neighbors
     * @param features Query feature vector
     * @param neighbors Output array of neighbor indices
     * @param distances Output array of distances
     */
    void findKNearestNeighbors(const FeatureVector& features,
                               uint8_t* neighbors,
                               float* distances);
    
    /**
     * Vote for class based on neighbors
     * @param neighbors Array of neighbor indices
     * @param distances Array of distances to neighbors
     * @return Predicted class and confidence
     */
    ObjectClass vote(const uint8_t* neighbors, const float* distances);
};

#endif // OBJECT_CLASSIFIER_H
