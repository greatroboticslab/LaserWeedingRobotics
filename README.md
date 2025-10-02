# Autonomous Laser Weed Control System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)

## 🌾 Overview

This project presents a comprehensive **Autonomous Laser Weed Control System** that combines artificial intelligence, precision laser technology, and robotic navigation for automated agricultural weed management. The system uses deep learning-based computer vision to identify weeds and precisely targets them with laser technology, offering an eco-friendly alternative to chemical herbicides.

## 🎯 Key Features

- **🤖 AI-Powered Weed Detection**: YOLOv11-based deep learning model for real-time weed identification
- **🎯 Precision Laser Control**: High-accuracy laser targeting system with dual-motor control
- **🧭 Autonomous Navigation**: GPS and IMU-based navigation with real-time positioning
- **🎮 Multi-Interface Control**: Web interface, gamepad control, and manual operation modes
- **📡 Real-time Communication**: MQTT-based data exchange and monitoring
- **⚡ Edge Computing**: Optimized for Jetson Nano and Raspberry Pi deployment

- ## 🆕 Latest Updates (v2.0 - October 2025)
**Major enhancements to the predictive targeting system:**
- 🎮 FlySky remote control integration with 4 operation modes
- ⚡ Dual motor synchronized laser tracking  
- 🧠 Intelligent area filtering (91% reduction in false positives)
- 🎯 Two-phase static targeting system (AIMING → FIRING)
- 📡 ESP32 dual-core optimization (<10ms system latency)

See [Predictive Aim README](Laser_System/predictiveaim/) for detailed technical documentation.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Vision     │    │  Laser Control  │    │   Navigation    │
│   (Jetson Nano) │◄──►│   (ESP32 +      │◄──►│ (Raspberry Pi)  │
│                 │    │    Helios DAC)  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MQTT Communication Hub                      │
│                    (Central Coordinator)                       │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
lasergithub/
├── Laser_System/                    # Laser control and targeting
│   ├── calibration and test/        # System calibration tools
│   │   ├── calibration.py          # Laser-camera calibration
│   │   └── doublelasertest.py      # Dual laser testing
│   ├── predictiveaim/              # Predictive targeting
│   │   └── predictiveaim.py        # Motion prediction algorithms
│   └── lasercar_system（old_version）/ # Web interface & control
│       ├── app.py                  # Flask web application
│       ├── esp32_robot_control.ino # ESP32 firmware
│       ├── static/                 # Web assets
│       └── templates/              # HTML templates
├── Yolo/                           # AI vision system
│   ├── code/                       # Training and data management
│   │   ├── fine_tuning.py         # YOLOv11 model fine-tuning
│   │   └── weed_lable_update_download.py # Roboflow integration
│   └── finetuned_trinning_result/  # Training results and metrics
└── auto_navigation/                # Navigation and positioning
    ├── jetsonnano/                 # Jetson Nano deployment
    └── resberry pi/                # Raspberry Pi IMU/GPS
```

## 🔧 Hardware Requirements

### Core Components
- **Jetson Nano**: AI inference and computer vision processing
- **Raspberry Pi 4**: IMU/GPS data collection and MQTT communication
- **ESP32**: Motor control and laser system coordination
- **Helios Laser DAC**: High-precision laser positioning control
- **BerryIMU**: 9-axis IMU sensor for orientation and movement tracking
- **USB Camera**: Real-time video capture for weed detection

### Laser System
- Dual-motor galvanometer system for XY laser positioning
- Laser diode with adjustable power control
- Safety interlocks and emergency stop mechanisms

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/laser-weed-control.git
cd laser-weed-control

# Install Python dependencies
pip install ultralytics opencv-python flask paho-mqtt numpy scipy

# Install additional requirements for each subsystem
pip install -r requirements.txt
```

### 2. Hardware Configuration

#### Jetson Nano Setup
```bash
# Enable GPU performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Start Docker container for YOLO inference
sudo docker start yolov11n
sudo docker exec -it yolov11n /bin/bash
```

#### Raspberry Pi Setup
```bash
# Install IMU libraries
sudo apt-get install python3-dev python3-pip
pip3 install paho-mqtt gpsd-py3

# Enable I2C and GPS
sudo raspi-config
```

### 3. System Calibration

```bash
# Run laser-camera calibration
cd Laser_System/calibration\ and\ test/
python calibration.py

# Test dual laser system
python doublelasertest.py
```

### 4. Start the System

```bash
# Terminal 1: Start web interface
cd Laser_System/lasercar_system（old_version）/
python app.py

# Terminal 2: Start navigation system
cd auto_navigation/resberry\ pi/
python3 berryIMU_mqtt.py

# Terminal 3: Start AI inference (on Jetson Nano)
python3 weed4.py
```

## 🤖 AI Model Training

### Dataset Preparation
The system uses YOLOv11 for weed detection with custom dataset integration via Roboflow:

```bash
cd Yolo/code/
python weed_lable_update_download.py  # Download and manage datasets
python fine_tuning.py                 # Train custom model
```

### Model Performance
- **mAP@0.5**: Achieved high accuracy on custom weed dataset
- **Real-time Inference**: Optimized for edge deployment
- **Custom Classes**: Trained specifically for agricultural weed species

## 🎮 Control Interfaces

### Web Interface
- **Real-time Monitoring**: Live camera feed with detection overlays
- **Manual Control**: Direct laser positioning and power control
- **System Status**: Hardware health and performance metrics
- **Safety Controls**: Emergency stop and laser disable functions

### Gamepad Control
- **Dual Stick Control**: Precise manual positioning
- **Button Mapping**: Quick access to common functions
- **Safety Lockouts**: Prevents accidental laser activation

## 🛡️ Safety Features

- **Hardware Emergency Stop**: Physical kill switch for immediate shutdown
- **Software Interlocks**: Multiple layers of safety validation
- **Laser Power Limiting**: Configurable maximum power levels
- **Motion Detection**: Automatic shutdown on unexpected movement
- **Timeout Protection**: Automatic shutdown after inactivity

## 📊 Performance Metrics

- **Detection Accuracy**: >95% weed identification rate
- **Targeting Precision**: Sub-millimeter laser positioning
- **Processing Speed**: Real-time operation at 20+ FPS
- **Coverage Rate**: Configurable scan patterns and speeds

## 🔬 Research Applications

This system serves as a platform for research in:
- **Precision Agriculture**: Sustainable farming practices
- **Computer Vision**: Agricultural object detection and classification
- **Robotics**: Autonomous navigation and manipulation
- **Laser Applications**: Non-contact material processing

## 🤝 Contributing

We welcome contributions from the research community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style and standards
- Testing procedures
- Documentation requirements
- Submission process

## 📚 Publications & Citations

If you use this system in your research, please cite:
```bibtex
@misc{laser_weed_control_2024,
  title={Autonomous Laser Weed Control System},
  author={[Your Name]},
  year={2024},
  howpublished={\url{https://github.com/yourusername/laser-weed-control}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv11 framework and community support
- **Roboflow**: Dataset management and annotation tools
- **OpenCV**: Computer vision processing capabilities
- **Flask**: Web interface framework

## 📞 Contact & Support

For questions, suggestions, or collaboration opportunities:
- **Issues**: [GitHub Issues](https://github.com/yourusername/laser-weed-control/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/laser-weed-control/discussions)
- **Email**: your.email@institution.edu

---


**⚠️ Safety Notice**: This system uses laser technology. Always follow proper safety protocols and local regulations when operating laser equipment. Ensure proper eye protection and safety barriers are in place during operation.
