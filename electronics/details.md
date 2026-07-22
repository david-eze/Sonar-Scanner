### ESP32-S3 Board
- ESP32-S3-DevKitC-1 or equivalent
- Dual-core Xtensa LX7 processor
- PSRAM support (optional but recommended)

### Sensors
- **Ultrasonic Sensor**: HC-SR04 or JSN-SR04T
  - Range: 2cm to 400cm
  - Trigger/Echo interface
  - Operating voltage: 5V DC
  - Current draw: 15mA

### Motor
- **Servo Motor**: Standard PWM servo (SG90, MG996R, etc.)
  - Range: 0-180 degrees
  - PWM control at 50Hz
  - Operating voltage: 5V DC
  - Current draw: 500mA peak (stall)

### Pin Configuration
```
TRIGGER_PIN  -> GPIO5  (Output to ultrasonic trigger)
ECHO_PIN     -> GPIO18 (Input from ultrasonic echo, interrupt-driven)
SERVO_PIN    -> GPIO16 (PWM output to servo)
UART0        -> USB/Serial (921600 baud telemetry)
VCC          -> 5V (servo/sensor) and 3.3V (ESP32)
GND          -> Common ground
```

## Electronics Design

### Power Supply System
The system requires a dual-voltage power supply:
- **5V Rail**: Powers ultrasonic sensor and servo motor
- **3.3V Rail**: Powers ESP32-S3 microcontroller

**Recommended Power Supply**: 5V 1A external power adapter with voltage regulator

**Voltage Regulation**:
- Use LM1117-3.3 or AMS1117-3.3 LDO regulator for 5V→3.3V conversion
- Input capacitors: 10μF electrolytic + 0.1μF ceramic
- Output capacitors: 10μF electrolytic + 0.1μF ceramic
- Ensure regulator can handle 500mA continuous current

### Circuit Description

#### ESP32-S3 Power Section
- **3.3V Supply**: Connected to ESP32 3V3 pin
- **Decoupling**: 0.1μF ceramic capacitor near each VCC pin
- **EN Pin**: Pulled high to enable ESP32
- **Reset**: 10kΩ pull-up resistor with tactile switch to ground

#### Ultrasonic Sensor Interface
- **Trigger Pin (GPIO5)**: ESP32 output → HC-SR04 Trig
- **Echo Pin (GPIO18)**: HC-SR04 Echo → ESP32 input with interrupt
- **Power**: HC-SR04 VCC → 5V rail
- **Ground**: Common ground with ESP32
- **Signal Conditioning**: 1kΩ series resistor on echo line for ESD protection
- **Pull-down**: 10kΩ resistor on echo pin to prevent floating

#### Servo Motor Control
- **PWM Signal (GPIO16)**: ESP32 LEDC channel → Servo signal line
- **Power**: Servo VCC → 5V rail (separate from ESP32 to avoid brownout)
- **Ground**: Common ground
- **Decoupling**: 100μF electrolytic capacitor near servo power pins
- **Note**: Servo current spikes (up to 1A) require robust 5V supply

#### Serial Communication
- **UART0**: USB interface for programming and telemetry
- **TX/RX**: Connected to USB-serial bridge on ESP32 dev board
- **Baud Rate**: 921600 for high-speed telemetry
- **Flow Control**: Not required (JSON protocol)

#### Status LEDs (Optional)
- **Power LED**: 3.3V → 220Ω resistor → LED → GND
- **Sensor LED**: GPIO controlled (active during measurement)
- **Telemetry LED**: GPIO controlled (blinks during data transmission)

### Noise Reduction Techniques
- **Ground Plane**: Use solid ground plane in PCB design
- **Decoupling**: 0.1μF capacitors at each IC power pin
- **Separation**: Keep high-current servo traces away from sensitive analog signals
- **Shielding**: Consider shielded cable for ultrasonic sensor if long distance
- **Filtering**: RC low-pass filter on servo PWM line if needed

### Protection Components
- **ESD Protection**: TVS diodes on external connectors
- **Reverse Polarity**: Schottky diode on power input
- **Overcurrent**: PTC fuse on 5V rail (1A hold current)
- **Voltage Clamping**: Zener diodes on GPIO lines exposed to external connections

### PCB Layout Recommendations
- **Component Placement**: ESP32 centrally, connectors at edges
- **Trace Width**: Power traces ≥ 0.3mm, signal traces ≥ 0.15mm
- **Via Usage**: Multiple vias for ground connections
- **Thermal Relief**: Large copper areas for heat dissipation
- **Mounting**: 4 corner holes with 3mm diameter for M3 screws

### Bill of Materials (BOM)
| Component | Value | Package | Quantity | Notes |
|-----------|-------|---------|----------|-------|
| ESP32-S3 | DevKitC-1 | Module | 1 | With USB-C |
| HC-SR04 | Ultrasonic | Module | 1 | Or JSN-SR04T |
| Servo Motor | SG90/MG996R | - | 1 | PWM servo |
| Voltage Regulator | LM1117-3.3 | TO-220 | 1 | 5V→3.3V |
| Capacitor | 10μF | Electrolytic | 2 | Power filtering |
| Capacitor | 0.1μF | Ceramic | 6 | Decoupling |
| Capacitor | 100μF | Electrolytic | 1 | Servo power |
| Resistor | 10kΩ | 0805 | 2 | Pull-ups |
| Resistor | 1kΩ | 0805 | 1 | ESD protection |
| Resistor | 220Ω | 0805 | 1 | LED current limit |
| LED | Green | 0805 | 1 | Power indicator |
| TVS Diode | SMBJ5.0A | SMB | 2 | ESD protection |
| PTC Fuse | 1A | 1812 | 1 | Overcurrent protection |
| Terminal Block | 2-pin | - | 2 | External connections |

### Power Budget Analysis
| Component | Voltage |Current (typical) | Current (peak) | Power |
|-----------|---------|------------------|----------------|-------|
| ESP32-S3 | 3.3V | 80mA | 300mA | 0.26W |
| Ultrasonic | 5V | 15mA | 15mA | 0.075W |
| Servo | 5V | 100mA | 500mA | 0.5W |
| **Total** | - | **195mA** | **815mA** | **0.84W** |

**Recommended Supply**: 5V 1A minimum, 5V 2A for safety margin
