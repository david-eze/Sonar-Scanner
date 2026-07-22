#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <driver/ledc.h>
#include <ArduinoJson.h>
#include <math.h>
#include "extended_kalman_filter.h"
#include "object_classifier.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * AI-Enhanced Sonar & Spatial Occupancy Scanner
 * Dual-Core ESP32-S3 FreeRTOS Architecture
 * 
 * Core 0 (APP_CPU): High-frequency sensor acquisition with EKF filtering
 * Core 1 (PRO_CPU): Motor control and telemetry streaming
 * 
 * Hardware Configuration:
 * - Ultrasonic Sensor: HC-SR04 (Trigger: GPIO5, Echo: GPIO18)
 * - Servo Motor: PWM on GPIO16 (LEDC channel)
 * - Serial: UART0 at 921600 baud for telemetry
 */

// ==================== HARDWARE CONFIGURATION ====================
#define TRIGGER_PIN      5
#define ECHO_PIN         18
#define SERVO_PIN        16

#define SERVO_PWM_FREQ   50      // 50Hz for standard servo
#define SERVO_PWM_RES    16      // 16-bit resolution
#define SERVO_MIN_PULSE  500     // 0.5ms pulse (0 degrees)
#define SERVO_MAX_PULSE  2500    // 2.5ms pulse (180 degrees)

#define SERIAL_BAUD      921600
#define UART_PORT        UART_NUM_0

// ==================== SYSTEM PARAMETERS ====================
#define SENSOR_SAMPLE_RATE_HZ    100    // 100Hz sensor sampling
#define SWEEP_PERIOD_MS          2000   // 2 seconds for full sweep
#define TELEMETRY_RATE_HZ        50     // 50Hz telemetry output

#define MAX_DISTANCE_M          4.0f    // Maximum measurable distance
#define SPEED_OF_SOUND          343.0f  // Speed of sound in m/s

// ==================== SHARED DATA STRUCTURES ====================
// Thread-safe data structure for inter-core communication
struct SensorData {
    float distance;           // Filtered distance (meters)
    float velocity;           // Estimated velocity (m/s)
    float confidence;         // Measurement confidence (0-1)
    float angle;              // Current servo angle (degrees)
    uint32_t timestamp;       // Microsecond timestamp
    ObjectClassifier::ObjectClass objectClass;
    float classificationConfidence;
};

struct MotorCommand {
    float targetAngle;        // Target angle (degrees)
    float sweepAmplitude;     // Sweep amplitude (degrees)
    float sweepFrequency;     // Sweep frequency (Hz)
    bool sweepEnabled;        // Enable sinusoidal sweep
};

// Shared data with mutex protection
SensorData currentSensorData;
MotorCommand motorCommand;
SemaphoreHandle_t sensorDataMutex;
SemaphoreHandle_t motorCommandMutex;

// ==================== GLOBAL OBJECTS ====================
ExtendedKalmanFilter ekf(0.1f, 0.3f);  // EKF with tuned parameters
ObjectClassifier classifier(3);        // k-NN classifier with k=3

// ==================== INTERRUPT HANDLERS ====================
volatile uint32_t echoStartTime = 0;
volatile uint32_t echoEndTime = 0;
volatile bool echoReceived = false;
volatile bool measurementReady = false;
float rawDistance = 0.0f;
float rawConfidence = 1.0f;

void IRAM_ATTR echoISR() {
    uint32_t currentTime = micros();
    
    if (digitalRead(ECHO_PIN) == HIGH) {
        // Rising edge - echo start
        echoStartTime = currentTime;
    } else {
        // Falling edge - echo end
        echoEndTime = currentTime;
        echoReceived = true;
    }
}

// ==================== CORE 0: SENSOR ACQUISITION TASK ====================
void sensorAcquisitionTask(void* pvParameters) {
    (void)pvParameters;
    
    Serial.println("[Core 0] Sensor Acquisition Task Started");
    
    const TickType_t taskDelay = pdMS_TO_TICKS(1000 / SENSOR_SAMPLE_RATE_HZ);
    uint32_t lastUpdateTime = micros();
    
    while (1) {
        uint32_t currentTime = micros();
        float dt = (currentTime - lastUpdateTime) / 1000000.0f;
        lastUpdateTime = currentTime;
        
        // Trigger ultrasonic measurement
        digitalWrite(TRIGGER_PIN, HIGH);
        delayMicroseconds(10);  // 10μs trigger pulse
        digitalWrite(TRIGGER_PIN, LOW);
        
        // Wait for echo with timeout
        echoReceived = false;
        uint32_t timeout = currentTime + 30000;  // 30ms timeout
        
        while (!echoReceived && micros() < timeout) {
            vTaskDelay(pdMS_TO_TICKS(1));
        }
        
        if (echoReceived) {
            // Calculate distance from echo time
            uint32_t echoDuration = echoEndTime - echoStartTime;
            rawDistance = (echoDuration * SPEED_OF_SOUND) / 2000000.0f;  // Divide by 2 for round-trip
            
            // Calculate confidence based on signal quality
            // Shorter echo times typically indicate stronger returns
            if (echoDuration < 1000) {
                rawConfidence = 1.0f;
            } else if (echoDuration < 5000) {
                rawConfidence = 0.9f;
            } else if (echoDuration < 10000) {
                rawConfidence = 0.7f;
            } else {
                rawConfidence = 0.5f;
            }
            
            // Validate measurement range
            if (rawDistance > MAX_DISTANCE_M || rawDistance < 0.02f) {
                rawDistance = 0.0f;
                rawConfidence = 0.0f;
            }
            
            measurementReady = true;
        } else {
            // Timeout - no valid measurement
            rawDistance = 0.0f;
            rawConfidence = 0.0f;
        }
        
        // Apply EKF filtering if valid measurement
        if (measurementReady && rawConfidence > 0.3f) {
            // Check for outliers using EKF
            if (!ekf.isOutlier(rawDistance, 4.0f)) {
                ekf.predict(dt);
                ekf.update(rawDistance, rawConfidence);
            }
        } else {
            // Just predict without update (dead reckoning)
            ekf.predict(dt);
        }
        
        // Update shared sensor data
        if (xSemaphoreTake(sensorDataMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            currentSensorData.distance = ekf.getDistance();
            currentSensorData.velocity = ekf.getVelocity();
            currentSensorData.confidence = rawConfidence;
            currentSensorData.timestamp = currentTime;
            
            // Get current motor angle
            if (xSemaphoreTake(motorCommandMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
                // Read current servo position from hardware
                // For now, use target angle as approximation
                currentSensorData.angle = motorCommand.targetAngle;
                xSemaphoreGive(motorCommandMutex);
            }
            
            xSemaphoreGive(sensorDataMutex);
        }
        
        vTaskDelay(taskDelay);
    }
}

// ==================== CORE 1: MOTOR CONTROL & TELEMETRY TASK ====================
void motorControlTask(void* pvParameters) {
    (void)pvParameters;
    
    Serial.println("[Core 1] Motor Control & Telemetry Task Started");
    
    const TickType_t taskDelay = pdMS_TO_TICKS(1000 / TELEMETRY_RATE_HZ);
    uint32_t sweepStartTime = micros();
    
    // Initialize motor command
    motorCommand.sweepAmplitude = 90.0f;  // ±90 degrees
    motorCommand.sweepFrequency = 0.5f;   // 0.5Hz sweep
    motorCommand.sweepEnabled = true;
    
    // Feature extraction history for classification
    float featureHistory[10];
    float confidenceHistory[10];
    uint8_t featureIndex = 0;
    uint8_t featureCount = 0;
    
    while (1) {
        uint32_t currentTime = micros();
        
        // Sinusoidal sweep control
        if (xSemaphoreTake(motorCommandMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            if (motorCommand.sweepEnabled) {
                float elapsed = (currentTime - sweepStartTime) / 1000000.0f;
                float sweepPhase = 2.0f * M_PI * motorCommand.sweepFrequency * elapsed;
                motorCommand.targetAngle = motorCommand.sweepAmplitude * sinf(sweepPhase);
            }
            
            // Convert angle to PWM duty cycle
            // Map -90 to +90 degrees to SERVO_MIN_PULSE to SERVO_MAX_PULSE
            float normalizedAngle = (motorCommand.targetAngle + 90.0f) / 180.0f;
            uint32_t pulseWidth = SERVO_MIN_PULSE + 
                                  (uint32_t)(normalizedAngle * (SERVO_MAX_PULSE - SERVO_MIN_PULSE));
            
            // Set PWM duty cycle
            uint32_t duty = (pulseWidth * (1 << SERVO_PWM_RES)) / (1000000 / SERVO_PWM_FREQ);
            ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, duty);
            ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
            
            xSemaphoreGive(motorCommandMutex);
        }
        
        // Object classification
        if (xSemaphoreTake(sensorDataMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            // Update feature history
            featureHistory[featureIndex] = currentSensorData.distance;
            confidenceHistory[featureIndex] = currentSensorData.confidence;
            featureIndex = (featureIndex + 1) % 10;
            if (featureCount < 10) featureCount++;
            
            // Extract features and classify
            if (featureCount >= 3) {
                ObjectClassifier::FeatureVector features = 
                    classifier.extractFeatures(featureHistory, confidenceHistory, 
                                            featureCount, 0.01f);
                
                ObjectClassifier::ObjectClass cls = classifier.classify(features);
                currentSensorData.objectClass = cls;
                currentSensorData.classificationConfidence = classifier.getConfidence();
            }
            
            xSemaphoreGive(sensorDataMutex);
        }
        
        // Stream telemetry over serial
        if (xSemaphoreTake(sensorDataMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            DynamicJsonDocument doc(256);
            
            doc["t"] = (unsigned long)currentSensorData.timestamp;
            doc["d"] = currentSensorData.distance;      // Distance (m)
            doc["v"] = currentSensorData.velocity;      // Velocity (m/s)
            doc["a"] = currentSensorData.angle;         // Angle (degrees)
            doc["c"] = currentSensorData.confidence;    // Confidence (0-1)
            doc["oc"] = (int)currentSensorData.objectClass;  // Object class
            doc["cc"] = currentSensorData.classificationConfidence;  // Classification confidence
            
            serializeJson(doc, Serial);
            Serial.println();
            
            xSemaphoreGive(sensorDataMutex);
        }
        
        vTaskDelay(taskDelay);
    }
}

// ==================== SETUP ====================
void setup() {
    Serial.begin(SERIAL_BAUD);
    Serial.println("AI-Enhanced Sonar Scanner Initializing...");
    
    // Initialize mutexes
    sensorDataMutex = xSemaphoreCreateMutex();
    motorCommandMutex = xSemaphoreCreateMutex();
    
    if (sensorDataMutex == NULL || motorCommandMutex == NULL) {
        Serial.println("Error: Failed to create mutexes");
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }
    
    // Initialize GPIO pins using Arduino framework
    pinMode(TRIGGER_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    digitalWrite(TRIGGER_PIN, LOW);
    
    // Attach interrupt for echo pin
    attachInterrupt(digitalPinToInterrupt(ECHO_PIN), echoISR, CHANGE);
    
    // Initialize servo PWM using LEDC
    ledc_timer_config_t timer_conf;
    timer_conf.speed_mode = LEDC_LOW_SPEED_MODE;
    timer_conf.duty_resolution = (ledc_timer_bit_t)SERVO_PWM_RES;
    timer_conf.timer_num = LEDC_TIMER_0;
    timer_conf.freq_hz = SERVO_PWM_FREQ;
    timer_conf.clk_cfg = LEDC_AUTO_CLK;
    ledc_timer_config(&timer_conf);
    
    ledc_channel_config_t channel_conf;
    channel_conf.gpio_num = SERVO_PIN;
    channel_conf.speed_mode = LEDC_LOW_SPEED_MODE;
    channel_conf.channel = LEDC_CHANNEL_0;
    channel_conf.intr_type = LEDC_INTR_DISABLE;
    channel_conf.timer_sel = LEDC_TIMER_0;
    channel_conf.duty = 0;
    channel_conf.hpoint = 0;
    ledc_channel_config(&channel_conf);
    
    // Initialize shared data
    currentSensorData.distance = 0.0f;
    currentSensorData.velocity = 0.0f;
    currentSensorData.confidence = 0.0f;
    currentSensorData.angle = 0.0f;
    currentSensorData.timestamp = 0;
    currentSensorData.objectClass = ObjectClassifier::UNKNOWN;
    currentSensorData.classificationConfidence = 0.0f;
    
    motorCommand.targetAngle = 0.0f;
    motorCommand.sweepAmplitude = 90.0f;
    motorCommand.sweepFrequency = 0.5f;
    motorCommand.sweepEnabled = true;
    
    // Create FreeRTOS tasks on specific cores
    // Core 0 (APP_CPU): Sensor acquisition
    xTaskCreatePinnedToCore(
        sensorAcquisitionTask,
        "SensorAcquisition",
        4096,
        NULL,
        2,  // Priority
        NULL,
        0   // Core 0
    );
    
    // Core 1 (PRO_CPU): Motor control and telemetry
    xTaskCreatePinnedToCore(
        motorControlTask,
        "MotorControl",
        4096,
        NULL,
        1,  // Priority
        NULL,
        1   // Core 1
    );
    
    Serial.println("System Initialized - Dual-Core Active");
}

// ==================== LOOP ====================
void loop() {
    // Main loop is empty - all work done in FreeRTOS tasks
    vTaskDelay(pdMS_TO_TICKS(1000));
}
