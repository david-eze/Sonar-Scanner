# Sonar Scanner CAD

## Assembly Description

The yellow arm descending from the sensor's dual transducers (green HC-SR04), through the lid opening, down to the red servo mount bracket with the yellow SG90 body inside. The green PCBs (Custom PCB left, ESP32 right) are visible inside the enclosure. The M3 corner hardware (grey) is at all four corners.

| Component | Colour | Description |
| :--- | :--- | :--- |
| LowerShell | 🔵 Blue | 100×80×60mm shell, shelled 2.5mm, filleted corners |
| UpperLid | 🔵 Blue | 100×80×3mm lid with vent slots, tower opening, M3 holes |
| BasePlate | ⬜ Grey | Internal floor plate with M3 clearance holes |
| Standoffs | ⬜ White | 8× hex standoffs for PCB + ESP32 (5mm height) |
| CustomPCB | 🟢 Green | 50×50mm PCB with voltage regulator + terminal block detail |
| ESP32S3 | 🟢 Green | DevKitC-1 with module, USB-C port, antenna, status LEDs |
| ServoMountTower | 🔴 Red | U-bracket with servo pocket + mounting flanges |
| ServoBody_SG90 | 🟡 Yellow | SG90 body with flanges + cross-arm horn |
| ScanningArm | 🟡 Yellow | Vertical stem + sensor bracket at 78mm height |
| SensorHCSR04 | 🟢 Green | 45×20mm PCB with 2 transducer cylinders |
| CableRouting | ⬛ Black | U-channel trunking along right inner wall |
| AlignmentPins | ⬜ White | 4× corner alignment pins at lid mating surface |
| M3_Screws | 🔘 Aluminium | 8× cap screws (4 base mount + 4 lid) |
| LabelPlate | ⬜ White | Front-face label area |

---

## Key Design Features Implemented

* **Ventilation** — 5 tall slots on right wall + 8 slots on lid
* **USB-C access** — slot on front wall aligned to ESP32 port
* **Cable glands** — 2× openings on left wall with strain relief channels
* **Tripod mount** — 1/4"-20 boss on bottom exterior
* **Snap-fit tabs** — 4× retention ribs inside shell top edge
* **Alignment pins** — 4× corner posts for lid registration
* **Cable channel groove** — routed trough on floor
* **LED window** — transparent opening on front face
* **M3 corner holes** — through-bottom and lid for external mounting

---

## Bill of Materials

18 line items including 3D-printed parts, screws, standoffs, 1/4"-20 tripod insert, terminal blocks, JST connectors, and status LEDs.
