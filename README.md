# Unitree Go2 — System Setup & SDK2 Communication Guide

This repository provides a **system-level setup and communication stack**
for working with the **Unitree Go2** quadruped using:
- **Unitree SDK2 (Python)** for low-level robot control
- **CycloneDDS** for real-time communication
- **ROS 2** (Humble on PC, Foxy on robot) for tooling and integration
- **Wired and wireless networking**

> ⚠️ This README focuses on **installation, networking, and communication only**.  
> Simulation, control logic, and sim-to-real deployment are documented separately.

➡️ **Low-Level Simulation & Deployment Stack (Core Research Code):**  
[`example/go2/low_level/docs/README.md`](example/go2/low_level/docs/README.md)

---

## Scope & Intent

This repository contains **two clearly separated components**:

1. **Vendor code (Unitree SDK2 + examples)**  
   - Located under `unitree_sdk2py/` and parts of `example/`  
   - Copyright © Unitree Robotics  
   - Included for completeness and reference

2. **Custom sim-to-real research stack for Go2**  
   - Located under `example/go2/low_level/`  
   - Developed for reinforcement-learning–based locomotion research  
   - Actively maintained and documented in this repository

Only the **Go2 low-level stack** represents original research work.

---

## What This README Covers

### Covered
- ROS 2 installation (PC + robot)
- CycloneDDS installation and verification
- Network setup (Ethernet / WiFi)
- Unitree SDK2 (Python) installation
- Robot ↔ PC communication prerequisites

### Explicitly NOT Covered
- Training reinforcement-learning policies
- Isaac Gym / Isaac Lab training pipelines
- High-level autonomy or navigation
- Safety certification or production deployment
- Commercial or consumer use

> **Important architectural note:**  
> Low-level control and deployment **do not require ROS 2**.  
> The robot is controlled directly via **Unitree SDK2 + CycloneDDS**.  
> ROS 2 is used on the PC side only for tooling, integration, and future extensions.

---

## 1. Legacy Repository Key (Go2 Robots Only)

Some older Go2 system images rely on legacy APT repositories.

⚠️ **Skip this step unless package installation fails on the robot.**

```
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF42ED6FBAB17C654 \
  | sudo gpg --dearmor -o /etc/apt/keyrings/unitree-legacy.gpg
```

This step is required **only once per robot**.

---

## 2. ROS 2 Installation and Setup

### System Overview

| System    | OS           | ROS 2  |
| --------- | ------------ | ------ |
| PC        | Ubuntu 22.04 | Humble |
| Go2 Robot | Ubuntu 20.04 | Foxy   |

⚠️ **Important:**
ROS 2 Humble and Foxy **do not share a ROS graph**.
The Go2 is controlled via **CycloneDDS used directly by Unitree SDK2**,
not via native ROS topic communication between machines.

---

### 2.1 Install ROS 2 Humble (PC)

Skip if already installed.

```
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common curl -y
sudo add-apt-repository universe

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
| sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop -y
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Install development tools and CycloneDDS support:

```
sudo apt install -y \
  python3-rosdep \
  python3-colcon-common-extensions \
  python3-argcomplete \
  libcyclonedds0 \
  ros-humble-rmw-cyclonedds-cpp

sudo rosdep init
rosdep update
```

Force CycloneDDS as the ROS middleware:

```
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc
source ~/.bashrc
```

---

### 2.2 Install ROS 2 Foxy (Robot)

Skip if already installed.

```
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common curl -y
sudo add-apt-repository universe

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
| sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-foxy-desktop -y
echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

### 2.3 Verify ROS Installation

```
ros2 topic list
```

This only verifies the local ROS environment.

---

## 3. Wireless Robot Connection

### ⚠️ Warning

The following steps **manually control WiFi** and may bypass NetworkManager.
Use them only if standard network tools are unavailable.

---

### 3.1 Preferred Method (If NetworkManager Is Available)

```
nmcli device wifi list
nmcli device wifi connect "<SSID>" password "<PASSWORD>"
```

---

### 3.2 Manual WiFi Configuration (Fallback)

```
sudo cp /etc/wpa_supplicant/wpa_supplicant.conf \
        /etc/wpa_supplicant/wpa_supplicant.conf.bak
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
```

```ini
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=IN

network={
    ssid="YOUR_WIFI_NAME"
    psk="YOUR_WIFI_PASSWORD"
}
```

Restart WiFi:

```
sudo pkill wpa_supplicant
sudo ip link set wlan0 down
sudo ip link set wlan0 up
sudo wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf
sudo dhclient -v wlan0
```

Verify:

```
ip a show wlan0
ping -c 3 8.8.8.8
```

> Note: Manual configuration may not persist across reboots
> depending on the robot image.

---

### 3.3 SSH Into the Robot

```
ssh unitree@<ROBOT_IP>
```

Find the IP using:

```
ip a
```

---

## 4. Unitree SDK2 (Python) Installation

### Requirements

* Ubuntu 20.04 / 22.04
* Python **3.8 – 3.10**
* CycloneDDS **0.10.x**

### Installation

```
git clone https://github.com/<your-org>/sim2real_unitree_sdk2py.git
cd sim2real_unitree_sdk2py
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 5. CycloneDDS (Only If Required)

Only follow this section if DDS discovery or import errors occur.

```
pip uninstall cyclonedds -y
rm -rf ~/cyclonedds
```

```
sudo apt install -y git cmake build-essential python3-dev libssl-dev
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install -j$(nproc)
```

```
export CYCLONEDDS_HOME=~/cyclonedds/install
export CMAKE_PREFIX_PATH=$CYCLONEDDS_HOME:$CMAKE_PREFIX_PATH
```

Reinstall SDK:

```
cd ~/sim2real_unitree_sdk2py
pip install -e .
```

Verify:

```
python - <<EOF
import cyclonedds
print(cyclonedds.__version__)
EOF
```

---