# Intelligent Robotics Systems Practice Module Group Project Readme
---

## SECTION 1 : PROJECT TITLE
## Aerial filming with synchronized drones using Reinforcement Learning
  
---
## SECTION 2 : EXECUTIVE SUMMARY

Today, more than 50% of News Agencies have Drones Team that performs the coverage of ‘live’ events. Typically these events do not have the luxury of having retakes, hence multiple drones are required for taking multiple vantage points at the same time. A large amount of expertise and coordination required and thus there is a need for Autonomous Flight Formation for multiple drones that can assist the human operators.

This project sets out to explore Reinforcement Learning based approaches to solve Real-time online inference drone flight in formation while tracking target using Camera and GPS sensors. Through the use of calibrated setup of a single target drone with multiple follower drones, the team explores the the effectiveness of Single Multi-Inputs Agent Reinforcement Learning vs Multiple Single-Input Agents Reinforcement Learning. The team demonstrated the results of the models trained in a pre-recorded simulation video and then presented their findings and observations in the accompanied report.

---
## SECTION 3 : TEAM MEMBERS
Members  : Kenneth Goh, Raymond Ng, Wong Yoke Keong

---
## SECTION 4 : USAGE GUIDE

### (A) System Requirements
1. OS: Ubuntu 18.04.3 LTS or Windows 10 + Windows Subsystem for Linux 1
2. Python: 3.6.7 (Anaconda or Miniconda distribution preferred)
3. Microsoft AirSim (v1.2.2 on Windows 10 or 1.2.0 on Ubuntu)
4. GPU (Discrete GPU preferred for running environment, playing simulations and training)
5. For Ubuntu Setup:
   - Docker
   - nvidia-docker
6. For cloud training:
    - Google Cloud Platform

### (B) Downloading Code
1. Clone this project: `git clone https://github.com/raymondng76/IRS-Practice-Module-Dev.git`
2. Change Directory: `cd IRS-Practice-Module-Dev`
3. Follow further instructions below

### (C) AirSim Environment Install/Run

#### Windows 10
1. Download and unzip your preferred environment from the [AirSim release 1.2.2 page](https://github.com/microsoft/AirSim/releases/tag/v.1.2.2)
2. Run the AirSim environment by double-clicking on `run.bat`

#### AirSim docker on Ubuntu 18.04
1. Please refer to [airsim_docker_local_install_readme.md](airsim_docker_local_install_readme.md) for details on install

### (D) Python Environment and Dependencies
1. Create new conda environment: `conda create -n airsim python=3.6.7`
2. Switch environment: `conda activate airsim`
3. Install dependencies
   - Using pip: `pip install -r requirements.txt`

### (E) Loading of Model Weights
1. Ensure python dependencies have been installed. Then execute the below commands
    - Execute `gdown 'https://drive.google.com/uc?id=1ciGqwUpfNPQu_Ua7cowU8mDIXOG_9kkf'`
    - Unzip the weights: `unzip Final_Weights_Models.zip`
    - Note that the file is very large (48MB) and downloading over mobile is not recommended
2. Copy YOLOv3 model weights to `IRS-Practice-Module-Dev` main directory
    - `cp -r Final_Weights_Models/Yolov3_drone_weights/ IRS-Practice-Module-Dev/weights`
3. Copy desired RL model weights from different iterations to `IRS-Practice-Module-Dev/code` sub-directory
    - e.g. copy of RDQN Single Model, 3rd Iteration: `cp -r Final_Weights_Models/RDQN_Single_Model/3rd_Iteration/* IRS-Practice-Module-Dev/code`

### (F) Running the simulation (Supported in Local only)
1. Ensure the AirSim environment is running
2. To run the simulations for the selected models, execute `python <model>.py --play --load_model`
3. To stop the simulation press `Ctrl-c`

### (G) Training the RL Models

#### Local Training
1. Ensure the AirSim environment is running
2. To train the models from scratch, execute `python <model>.py --verbose`. Options include
   - `rdqn.py`
   - `rdqn_triple_model.py`
   - `rddpg_triple_model.py`
3. To resume training, execute `python <models>.py --verbose --load_model`
3. To stop the training press `Ctrl-c`

#### Training on Google Cloud Platform
1. Please refer to [gcp_training_readme.md](gcp_training_readme.md) for details on setup and training on Google Cloud Platform VM.
2. Please note that playing the simulation is not recommended on VM due to display challenges.
---

## SECTION 5 : SIMULATION VIDEO DEMO

### Iteration 2
[![Iteration 2](http://img.youtube.com/vi/ZT0SEAQG_U0/0.jpg)](https://www.youtube.com/watch?v=ZT0SEAQG_U0 "Iteration 2")

### Iteration 3 (Best/RDQN Single)
[![Iteration 2](http://img.youtube.com/vi/OdLcRP5R0MQ/0.jpg)](https://www.youtube.com/watch?v=OdLcRP5R0MQ "Iteration 2")

### Iteration 4
[![Iteration 2](http://img.youtube.com/vi/aweLkL8Xr18/0.jpg)](https://www.youtube.com/watch?v=aweLkL8Xr18 "Iteration 4")

---
## SECTION 6 : ACKNOWLEDGEMENT AND REFERENCES

- Code is based on the efforts of Sung Hoon Hong: [sunghoonhong/AirsimDRL: Autonomous UAV Navigation without Collision using Visual Information in Airsim](https://github.com/sunghoonhong/AirsimDRL)
- Object Detection code is based on: [experiencor/keras-yolo3: Training and Detecting Objects with YOLO3](https://github.com/experiencor/keras-yolo3)
- Neural Network framework used: [tensorflow/tensorflow: An Open Source Machine Learning Framework for Everyone](https://github.com/tensorflow/tensorflow)
- Drone Simulation Environment is from: [microsoft/AirSim: Open source simulator for autonomous vehicles built on Unreal Engine / Unity, from Microsoft AI & Research](https://github.com/microsoft/AirSim)
- Additional Citations are in the report

---