# Autonomous Laser Weed Control System

A comprehensive autonomous laser-based weed management system combining artificial intelligence, precision laser technology, and robotic navigation for sustainable agriculture.

## 🚀 New Features (Latest Update)

### 🎮 FlySky Remote Control Integration
- **SwA**: Moving target mode (autonomous predictive targeting)
- **SwB**: Auto static detection mode (automatic stationary weed elimination)
- **SwC**: Emergency/Manual control mode (joystick laser control)
- **SwD**: Laser power toggle
- Real-time WiFi communication with ESP32
- Manual joystick control for laser positioning and power

### 🎯 Dual Motor Laser System
- Synchronized dual-laser tracking
- Independent calibration for each motor
- Both lasers follow the same predicted trajectory
- Enhanced coverage and redundancy

### 🧠 Intelligent Weed Detection
- **Area filtering** to prevent false detection of large grass patches
- Aspect ratio filtering for edge detection
- Configurable size thresholds (0.08% - 18% of frame)
- Advanced noise filtering for shaky movements and rough terrain

### 📡 ESP32 Dual-Core Optimization
- **Core 0**: WiFi communication and data streaming
- **Core 1**: Real-time motor control (non-blocking)
- Double-buffer system for lock-free data exchange
- 20ms update rate with <5ms latency

### 🎪 Static Targeting System
- **Two-phase operation**:
  1. AIMING phase (laser OFF, 0.5-3s)
  2. FIRING phase (laser ON, 5-25s)
- Automatic detection of stationary weeds
- Configurable timeout and duration

### ⚡ Enhanced Trajectory Prediction
- YOLO processing delay compensation (auto-measured)
- Fast trajectory method for quick response
- Speed scaling factor for accuracy tuning
- Multi-stage noise filtering

## 🏗️ System Architecture

```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ AI Vision       │ │ Laser Control   │ │ Navigation      │
│ (Jetson Nano)   │◄──►│ (ESP32 +        │◄──►│ (Raspberry Pi)  │
│ - YOLOv11       │ │   Helios DAC)   │ │ - GPS/IMU       │
│ - Prediction    │ │ - Dual Motor    │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────┐
│          MQTT Communication Hub                         │
│        + FlySky Remote Control (WiFi)                   │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
lasergithub/
├── Laser_System/                    # Laser control and targeting
│   ├── calibration and test/        # System calibration tools
│   │   ├── calibration.py           # Laser-camera calibration
│   │   └── doublelasertest.py       # Dual laser testing
│   ├── predictiveaim/               # 🆕 Predictive targeting (UPDATED)
│   │   ├── predictiveaim.py         # Enhanced dual-motor tracking
│   │   ├── 930test.py               # Latest version with remote control
│   │   └── 930esp32.txt             # ESP32 firmware for FlySky integration
│   └── lasercar_system（old_version）/  # Web interface & control
│       ├── app.py                   # Flask web application
│       ├── esp32_robot_control.ino  # ESP32 firmware (old version)
│       ├── static/                  # Web assets
│       └── templates/               # HTML templates
├── Yolo/                            # AI vision system
│   ├── code/                        # Training and data management
│   │   ├── fine_tuning.py           # YOLOv11 model fine-tuning
│   │   └── weed_lable_update_download.py  # Roboflow integration
│   └── finetuned_trinning_result/   # Training results and metrics
└── auto_navigation/                 # Navigation and positioning
    ├── jetsonnano/                  # Jetson Nano deployment
    └── resberry pi/                 # Raspberry Pi IMU/GPS
```

## ✨ Key Features

### AI-Powered Detection
- 🤖 YOLOv11-based deep learning model
- 🎯 >95% weed identification rate
- ⚡ Real-time operation at 20+ FPS
- 🧹 Area filtering to ignore large grass patches

### Precision Laser Control
- 🎪 Sub-millimeter laser positioning
- 🔄 Dual-motor galvanometer system
- 🎨 Configurable laser patterns (zigzag, etc.)
- 💪 Adjustable power control (0-255)

### Autonomous Navigation
- 🧭 GPS and IMU-based positioning
- 🗺️ Real-time trajectory prediction
- 🎢 Noise filtering for rough terrain
- ⏱️ YOLO delay compensation

### Multi-Interface Control
- 🎮 FlySky remote control (4 modes)
- 💻 Web interface
- ⌨️ Keyboard controls
- 🖱️ Mouse/slider for power adjustment

### Safety Features
- 🛑 Hardware emergency stop
- 🔒 Software interlocks
- ⚠️ Motion detection shutdown
- ⏲️ Timeout protection
- 🚨 Multi-layer safety validation

## 🔧 Hardware Components

| Component | Purpose |
|-----------|---------|
| **Jetson Nano** | AI inference and computer vision |
| **Raspberry Pi 4** | IMU/GPS data collection, MQTT |
| **ESP32** | Motor control, laser coordination, FlySky receiver |
| **Helios Laser DAC** (x2) | High-precision dual laser control |
| **BerryIMU** | 9-axis IMU sensor |
| **USB Camera** | Real-time video capture |
| **FlySky Remote** | Wireless control system |
| **RoboClaw** | Motor controller |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/greatroboticslab/LaserWeedingRobotics.git
cd LaserWeedingRobotics

# Install dependencies
pip install ultralytics opencv-python flask paho-mqtt numpy scipy keyboard

# Install additional requirements
pip install -r requirements.txt
```

### 2. ESP32 Setup (FlySky Integration)

```bash
# Upload ESP32 firmware
# File: Laser_System/predictiveaim/930esp32.txt
# 1. Open in Arduino IDE
# 2. Install libraries: IBusBM, RoboClaw
# 3. Configure WiFi credentials
# 4. Upload to ESP32

# Default IP: 192.168.1.104
# Port: 10001
```

### 3. Calibration

```bash
# Calibrate both laser motors
cd Laser_System/calibration\ and\ test/
python calibration.py

# Test dual laser system
python doublelasertest.py
```

### 4. Run Main System

```bash
cd Laser_System/predictiveaim/
python 930test.py
```

## 🎮 FlySky Remote Control Modes

| Switch | Mode | Description | Laser |
|--------|------|-------------|-------|
| **SwA** | Moving Target | Autonomous predictive targeting for moving weeds | Auto ON/OFF |
| **SwB** | Auto Static | Automatic detection and elimination of stationary weeds | Auto ON |
| **SwC** | Manual Control | Full joystick control of laser position and power | Manual |
| **SwD** | Laser Toggle | Toggle laser ON/OFF (works in manual mode) | Toggle |

### Manual Mode Controls
- **Left Stick (CH3)**: Laser power (0-100%)
- **Right Stick (CH1/CH2)**: Laser X/Y position
- **Emergency**: SwC immediately stops all operations

## ⌨️ Keyboard Controls

### Main Controls
- `T`: Toggle Auto Targeting
- `L`: Toggle Laser (manual mode only)
- `Y`: Toggle Laser Patterns
- `ESC`: Exit

### Timing Adjustments
- `UP/DOWN`: Observation time (0.3-3.0s)
- `LEFT/RIGHT`: Prediction duration (2.0-20.0s)
- `PgUp/PgDn`: Prediction delay (0.0-5.0s)
- `HOME/END`: YOLO delay compensation (0.5-2.0s)

### Area Filtering
- `1/2`: Max area threshold (5-40%)
- `3/4`: Max aspect ratio (2.0-8.0)

### Static Targeting
- `5/6`: Aiming duration (0.5-3.0s)
- `7/8`: Stationary timeout (2.0-10.0s)
- `9/0`: Firing duration (5.0-25.0s)
- `B`: Stop static targeting

### Noise Filtering
- `Q/A`: Filter strength (0.0-1.0)
- `W/S`: Smoothing window (1-10)
- `E/D`: Movement threshold (1.0-20.0)
- `R/F`: Outlier threshold (10-200)

### Other
- `U/I`: Min execution confidence
- `O/K`: Speed scaling factor
- `V`: Toggle region display
- `P`: Toggle prediction display
- `X`: Toggle noise stats

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Detection Accuracy | >95% |
| Targeting Precision | Sub-millimeter |
| Processing Speed | 20+ FPS |
| Response Time | <100ms (with remote) |
| YOLO Latency | ~1s (auto-compensated) |
| Update Rate | 40 Hz |

## 🔬 Advanced Features

### Area Filtering Algorithm
- Filters detections >18% of frame (grass patches)
- Filters extreme aspect ratios (grass edges)
- Minimum detection size: 0.08% of frame
- Prevents false positives on large areas

### Noise Filtering Pipeline
1. **Outlier Detection**: Rejects sudden jumps >50px
2. **Moving Average**: Weighted smoothing (5-point window)
3. **Velocity Filtering**: Consistency-based filtering
4. **Direction Tracking**: Maintains trajectory consistency

### Trajectory Prediction
- **Fast Method**: Start-end position (0.8s observation)
- **Average Method**: Full velocity analysis
- **Delay Compensation**: Auto-measured YOLO latency
- **Speed Scaling**: Adjustable prediction accuracy

### Static Targeting Workflow
```
Detection → Monitor (5s) → AIMING (1s, laser OFF) → FIRING (10s, laser ON) → Complete
```

## 📡 Communication Protocols

### ESP32 → Python (TCP)
```
DATA timestamp CH1:val CH2:val ... ST:steer TH:throttle ML:left MR:right MD:mode SW:switches
```

### Python → ESP32 (Serial)
```
ON        # Turn laser on
OFF       # Turn laser off
POWER 128 # Set power (0-255)
```

## 🛡️ Safety Features

1. **Hardware Emergency Stop**: SwC switch
2. **Automatic Laser Shutoff**: When no target detected
3. **Manual Mode Protection**: Laser OFF by default in manual
4. **Motion Detection**: Stops on unexpected movement
5. **Timeout Protection**: Auto-shutdown after inactivity
6. **Multi-layer Validation**: Software interlocks

## 📚 Research Applications

- Precision agriculture and sustainable farming
- Computer vision for agricultural robotics
- Real-time trajectory prediction algorithms
- Multi-sensor fusion and calibration
- Human-robot interaction via remote control

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style and standards
- Testing procedures
- Documentation requirements
- Submission process

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv11 framework
- **Roboflow**: Dataset management
- **OpenCV**: Computer vision library
- **FlySky**: Remote control hardware

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/greatroboticslab/LaserWeedingRobotics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/greatroboticslab/LaserWeedingRobotics/discussions)

---

## 🆕 Latest Updates (v2.0)

### Major Improvements
✅ FlySky remote control integration with 4 operation modes  
✅ Dual-motor synchronized laser tracking  
✅ Intelligent area filtering to prevent grass patch false positives  
✅ ESP32 dual-core optimization for real-time performance  
✅ Two-phase static targeting (aiming + firing)  
✅ Enhanced noise filtering for rough terrain  
✅ Automatic YOLO delay compensation  
✅ Manual joystick control for laser positioning  

### Bug Fixes
🐛 Fixed large grass patch false detection  
🐛 Improved trajectory prediction stability  
🐛 Reduced laser control latency  

### Coming Soon
🔮 GPS-based autonomous navigation  
🔮 Multi-robot coordination  
🔮 Advanced pattern optimization  
🔮 Machine learning for laser power optimization
