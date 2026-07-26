#ifndef OBJECT_CLASSIFIER_H
#define OBJECT_CLASSIFIER_H

#include <Arduino.h>
#include <math.h>
#include <cmath>
#include <limits>

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
    
    struct FeatureVector {
        float distance;
        float velocity;
        float variance;
        float amplitude;
        float gradient;
        float consistency;
    };
    
    struct TrainingSample {
        FeatureVector features;
        ObjectClass label;
    };
    
    ObjectClassifier(uint8_t k = 3);
    
    FeatureVector extractFeatures(const float* distances, 
                                 const float* confidences,
                                 uint8_t count,
                                 float dt);
    
    ObjectClass classify(const FeatureVector& features);
    
    static const char* getClassName(ObjectClass cls);
    
    float getConfidence() const { return lastConfidence; }
    
    void initializeTrainingData();
    
    void addTrainingSample(const TrainingSample& sample);
    
    void reset();

private:
    static const uint8_t MAX_TRAINING_SAMPLES = 50;
    static const uint8_t MAX_HISTORY = 10;
    
    uint8_t kNeighbors;
    TrainingSample trainingData[MAX_TRAINING_SAMPLES];
    uint8_t numTrainingSamples;
    float lastConfidence;
    
    float historyDistances[MAX_HISTORY];
    float historyConfidences[MAX_HISTORY];
    uint8_t historyIndex;
    uint8_t historyCount;
    
    float calculateDistance(const FeatureVector& a, const FeatureVector& b);
    
    void normalizeFeatures(FeatureVector& features);
    
    struct FeatureBounds {
        float min;
        float max;
    };
    FeatureBounds featureBounds[6];
    
    void findKNearestNeighbors(const FeatureVector& features,
                               uint8_t* neighbors,
                               float* distances);
    
    ObjectClass vote(const uint8_t* neighbors, const float* distances);
};

#endif // OBJECT_CLASSIFIER_H
