#include "object_classifier.h"

ObjectClassifier::ObjectClassifier(uint8_t k) : kNeighbors(k), numTrainingSamples(0), lastConfidence(0.0f), historyIndex(0), historyCount(0) {
    featureBounds[0] = {0.0f, 4.0f};
    featureBounds[1] = {-2.0f, 2.0f};
    featureBounds[2] = {0.0f, 0.5f};
    featureBounds[3] = {0.0f, 1.0f};
    featureBounds[4] = {-5.0f, 5.0f};
    featureBounds[5] = {0.0f, 1.0f};
    
    for (uint8_t i = 0; i < MAX_HISTORY; i++) {
        historyDistances[i] = 0.0f;
        historyConfidences[i] = 0.0f;
    }
    
    initializeTrainingData();
}

void ObjectClassifier::initializeTrainingData() {
    numTrainingSamples = 0;
    
    TrainingSample wall1 = {{2.0f, 0.0f, 0.02f, 0.9f, 0.0f, 0.95f}, WALL_FLAT};
    trainingData[numTrainingSamples++] = wall1;
    
    TrainingSample wall2 = {{1.5f, 0.01f, 0.03f, 0.85f, 0.02f, 0.92f}, WALL_FLAT};
    trainingData[numTrainingSamples++] = wall2;
    
    TrainingSample wall3 = {{3.0f, -0.01f, 0.025f, 0.88f, -0.01f, 0.94f}, WALL_FLAT};
    trainingData[numTrainingSamples++] = wall3;
    
    TrainingSample corner1 = {{1.8f, 0.1f, 0.15f, 0.6f, 0.5f, 0.5f}, CORNER_EDGE};
    trainingData[numTrainingSamples++] = corner1;
    
    TrainingSample corner2 = {{2.2f, -0.15f, 0.2f, 0.55f, -0.6f, 0.45f}, CORNER_EDGE};
    trainingData[numTrainingSamples++] = corner2;
    
    TrainingSample corner3 = {{1.0f, 0.08f, 0.18f, 0.5f, 0.4f, 0.48f}, CORNER_EDGE};
    trainingData[numTrainingSamples++] = corner3;
    
    TrainingSample moving1 = {{1.5f, 0.8f, 0.1f, 0.7f, 1.2f, 0.6f}, DYNAMIC_MOVING};
    trainingData[numTrainingSamples++] = moving1;
    
    TrainingSample moving2 = {{2.0f, -0.9f, 0.12f, 0.65f, -1.5f, 0.55f}, DYNAMIC_MOVING};
    trainingData[numTrainingSamples++] = moving2;
    
    TrainingSample moving3 = {{1.2f, 0.6f, 0.08f, 0.75f, 0.8f, 0.65f}, DYNAMIC_MOVING};
    trainingData[numTrainingSamples++] = moving3;
    
    TrainingSample human1 = {{1.8f, 0.05f, 0.08f, 0.3f, 0.1f, 0.4f}, HUMAN_SOFT};
    trainingData[numTrainingSamples++] = human1;
    
    TrainingSample human2 = {{2.5f, -0.03f, 0.1f, 0.25f, -0.05f, 0.35f}, HUMAN_SOFT};
    trainingData[numTrainingSamples++] = human2;
    
    TrainingSample human3 = {{1.3f, 0.08f, 0.09f, 0.35f, 0.12f, 0.42f}, HUMAN_SOFT};
    trainingData[numTrainingSamples++] = human3;
}

void ObjectClassifier::addTrainingSample(const TrainingSample& sample) {
    if (numTrainingSamples < MAX_TRAINING_SAMPLES) {
        trainingData[numTrainingSamples++] = sample;
    }
}

ObjectClassifier::FeatureVector ObjectClassifier::extractFeatures(const float* distances, 
                                                                   const float* confidences,
                                                                   uint8_t count,
                                                                   float dt) {
    FeatureVector features;
    
    if (count < 2) {
        features.distance = distances[0];
        features.velocity = 0.0f;
        features.variance = 0.0f;
        features.amplitude = confidences[0];
        features.gradient = 0.0f;
        features.consistency = 0.0f;
        return features;
    }
    
    features.distance = distances[count - 1];
    
    features.velocity = (distances[count - 1] - distances[count - 2]) / dt;
    
    float mean = 0.0f;
    for (uint8_t i = 0; i < count; i++) {
        mean += distances[i];
    }
    mean /= count;
    
    float variance = 0.0f;
    for (uint8_t i = 0; i < count; i++) {
        float diff = distances[i] - mean;
        variance += diff * diff;
    }
    features.variance = sqrtf(variance / count);
    
    float amplitudeSum = 0.0f;
    for (uint8_t i = 0; i < count; i++) {
        amplitudeSum += confidences[i];
    }
    features.amplitude = amplitudeSum / count;
    
    if (count >= 3) {
        features.gradient = (distances[count - 1] - distances[count - 3]) / (2.0f * dt);
    } else {
        features.gradient = features.velocity;
    }
    
    float maxVariance = 0.5f;
    features.consistency = 1.0f - (features.variance / maxVariance);
    if (features.consistency < 0.0f) features.consistency = 0.0f;
    
    return features;
}

ObjectClassifier::ObjectClass ObjectClassifier::classify(const FeatureVector& features) {
    if (numTrainingSamples == 0) {
        return UNKNOWN;
    }
    
    FeatureVector normalizedFeatures = features;
    normalizeFeatures(normalizedFeatures);
    
    uint8_t neighbors[kNeighbors];
    float distances[kNeighbors];
    findKNearestNeighbors(normalizedFeatures, neighbors, distances);
    
    ObjectClass predictedClass = vote(neighbors, distances);
    
    uint8_t classCounts[NUM_CLASSES] = {0};
    for (uint8_t i = 0; i < kNeighbors; i++) {
        classCounts[trainingData[neighbors[i]].label]++;
    }
    
    uint8_t maxCount = 0;
    for (uint8_t i = 0; i < NUM_CLASSES; i++) {
        if (classCounts[i] > maxCount) {
            maxCount = classCounts[i];
        }
    }
    
    lastConfidence = (float)maxCount / kNeighbors;
    
    return predictedClass;
}

float ObjectClassifier::calculateDistance(const FeatureVector& a, const FeatureVector& b) {
    float sum = 0.0f;
    
    float diff = a.distance - b.distance;
    sum += diff * diff;
    
    diff = a.velocity - b.velocity;
    sum += diff * diff;
    
    diff = a.variance - b.variance;
    sum += diff * diff;
    
    diff = a.amplitude - b.amplitude;
    sum += diff * diff;
    
    diff = a.gradient - b.gradient;
    sum += diff * diff;
    
    diff = a.consistency - b.consistency;
    sum += diff * diff;
    
    return sqrtf(sum);
}

void ObjectClassifier::normalizeFeatures(FeatureVector& features) {
    features.distance = (features.distance - featureBounds[0].min) / (featureBounds[0].max - featureBounds[0].min);
    features.velocity = (features.velocity - featureBounds[1].min) / (featureBounds[1].max - featureBounds[1].min);
    features.variance = (features.variance - featureBounds[2].min) / (featureBounds[2].max - featureBounds[2].min);
    features.amplitude = (features.amplitude - featureBounds[3].min) / (featureBounds[3].max - featureBounds[3].min);
    features.gradient = (features.gradient - featureBounds[4].min) / (featureBounds[4].max - featureBounds[4].min);
    features.consistency = (features.consistency - featureBounds[5].min) / (featureBounds[5].max - featureBounds[5].min);
    
    if (features.distance < 0.0f) features.distance = 0.0f;
    if (features.distance > 1.0f) features.distance = 1.0f;
    if (features.velocity < 0.0f) features.velocity = 0.0f;
    if (features.velocity > 1.0f) features.velocity = 1.0f;
    if (features.variance < 0.0f) features.variance = 0.0f;
    if (features.variance > 1.0f) features.variance = 1.0f;
    if (features.amplitude < 0.0f) features.amplitude = 0.0f;
    if (features.amplitude > 1.0f) features.amplitude = 1.0f;
    if (features.gradient < 0.0f) features.gradient = 0.0f;
    if (features.gradient > 1.0f) features.gradient = 1.0f;
    if (features.consistency < 0.0f) features.consistency = 0.0f;
    if (features.consistency > 1.0f) features.consistency = 1.0f;
}

void ObjectClassifier::findKNearestNeighbors(const FeatureVector& features,
                                             uint8_t* neighbors,
                                             float* distances) {
    for (uint8_t i = 0; i < kNeighbors; i++) {
        distances[i] = std::numeric_limits<float>::max();
        neighbors[i] = 0;
    }
    
    for (uint8_t i = 0; i < numTrainingSamples; i++) {
        FeatureVector normalizedTraining = trainingData[i].features;
        normalizeFeatures(normalizedTraining);
        
        float dist = calculateDistance(features, normalizedTraining);
        
        for (uint8_t j = 0; j < kNeighbors; j++) {
            if (dist < distances[j]) {
                for (uint8_t k = kNeighbors - 1; k > j; k--) {
                    distances[k] = distances[k - 1];
                    neighbors[k] = neighbors[k - 1];
                }
                distances[j] = dist;
                neighbors[j] = i;
                break;
            }
        }
    }
}

ObjectClassifier::ObjectClass ObjectClassifier::vote(const uint8_t* neighbors, const float* distances) {
    float classWeights[NUM_CLASSES] = {0.0f};
    
    for (uint8_t i = 0; i < kNeighbors; i++) {
        ObjectClass cls = trainingData[neighbors[i]].label;
        float weight = 1.0f / (distances[i] + 0.001f);
        classWeights[cls] += weight;
    }
    
    float maxWeight = 0.0f;
    ObjectClass predictedClass = UNKNOWN;
    
    for (uint8_t i = 0; i < NUM_CLASSES; i++) {
        if (classWeights[i] > maxWeight) {
            maxWeight = classWeights[i];
            predictedClass = (ObjectClass)i;
        }
    }
    
    return predictedClass;
}

const char* ObjectClassifier::getClassName(ObjectClass cls) {
    switch (cls) {
        case UNKNOWN: return "Unknown";
        case WALL_FLAT: return "Wall/Flat";
        case CORNER_EDGE: return "Corner/Edge";
        case DYNAMIC_MOVING: return "Dynamic/Moving";
        case HUMAN_SOFT: return "Human/Soft";
        default: return "Invalid";
    }
}

void ObjectClassifier::reset() {
    historyIndex = 0;
    historyCount = 0;
    lastConfidence = 0.0f;
    
    for (uint8_t i = 0; i < MAX_HISTORY; i++) {
        historyDistances[i] = 0.0f;
        historyConfidences[i] = 0.0f;
    }
}
