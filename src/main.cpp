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

#define TRIGGER_PIN      5
#define ECHO_PIN         18
#define SERVO_PIN        16

#define SERVO_PWM_FREQ   50
#define SERVO_PWM_RES    16
#define SERVO_MIN_PULSE  500
#define SERVO_MAX_PULSE  2500

#define SERIAL_BAUD      921600
#define UART_PORT        UART_NUM_0

#define SENSOR_SAMPLE_RATE_HZ    100
#define SWEEP_PERIOD_MS          2000
#define TELEMETRY_RATE_HZ        50

#define MAX_DISTANCE_M           4.0f
#define SPEED_OF_SOUND           343.0f

struct SensorData {
    float distance;
    float velocity;
    float confidence;
    float angle;
    uint32_t timestamp;
    ObjectClassifier::ObjectClass objectClass;
    float classificationConfidence;
};

struct MotorCommand {
    float targetAngle;
    float sweepAmplitude;
    float sweepFrequency;
    bool sweepEnabled;
};

SensorData currentSensorData;
MotorCommand motorCommand;
SemaphoreHandle_t sensorDataMutex;
SemaphoreHandle_t motorCommandMutex;

ExtendedKalmanFilter ekf(0.1f, 0.3f);
ObjectClassifier classifier(3);

volatile uint32_t echoStartTime = 0;
volatile uint32_t echoEndTime = 0;
volatile bool echoReceived = false;
volatile bool measurementReady = false;
float rawDistance = 0.0f;
float rawConfidence = 1.0f;

void IRAM_ATTR echoISR() {
    uint32_t currentTime = micros();
    
    if (digitalRead(ECHO_PIN) == HIGH) {
        echoStartTime = currentTime;
    } else {
        echoEndTime = currentTime;
        echoReceived = true;
    }
}

void sensorAcquisitionTask(void* pvParameters) {
    (void)pvParameters;
    
    Serial.println("[Core 0] Sensor Acquisition Task Started");
    
    const TickType_t taskDelay = pdMS_TO_TICKS(1000 / SENSOR_SAMPLE_RATE_HZ);
    uint32_t lastUpdateTime = micros();
    
    while (1) {
        uint32_t currentTime = micros();
        float dt = (currentTime - lastUpdateTime) / 1000000.0f;
        lastUpdateTime = currentTime;
        
        digitalWrite(TRIGGER_PIN, HIGH);
        delayMicroseconds(10);
        digitalWrite(TRIGGER_PIN, LOW);
        
        echoReceived = false;
        uint32_t timeout = currentTime + 30000;
        
        while (!echoReceived && micros() < timeout) {
            vTaskDelay(pdMS_TO_TICKS(1));
        }
        
        if (echoReceived) {
            uint3
