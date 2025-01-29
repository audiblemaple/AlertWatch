# Facial landmarks detection

## Abstract
In modern environments, whether in vehicles, offices, or other settings, maintaining alertness and preventing fatigue-related incidents has become a critical concern. "AlertWatch" presents an innovative solution utilizing advanced computer vision and hardware to monitor individuals and detect signs of drowsiness in real-time. This project leverages facial recognition and facial landmark detection to continuously assess the alertness levels of a person. Upon detecting signs of drowsiness, "AlertWatch" triggers immediate alerts to prevent potential accidents, lapses in productivity or even security concerns. This paper details the development process of "AlertWatch," focusing on its implementation of real-time monitoring through cameras and software algorithms, as well as outlining the development process and hardware options. "AlertWatch" enhances safety and efficiency and sets new standards for proactive alertness management through technology.

## Project Docs
* [Book](Capstone_Project_Docs/Part_1/AlertWatch.docx)
* [Presentation](Capstone_Project_Docs/Part_1/Alertwatch.pptx)


## Developer documentation 
[DEV docs](docs/index.html)

## Components
The system comprises 2 parts:
* [ECU / Management unit](#ecu--management-unit)
* [Detection unit](#detection-unit)


## ECU / Management unit
### Software
* [ECU web UI](ECU/frontend)
* [ECU backend](ECU/backend)
* [Voice recognition](voice_recognition)

### Training
TODO
### Testing
TODO

### Hardware
* [SolidRun Bedrock R7000](https://www.solid-run.com/industrial-computers/bedrock-r7000-edgeai/)


## Detection unit

### Software
* [Face detection](face_detection)
* [Face landmarks detection](face_landmarks_detection)
* [Hailo Components](hailo)

### Training
TODO

### Testing

```bash
cd face_landmarks_detection/testing/pytorch

## For face
python3 validate_opencv.py -m face 

## For eyes
python3 validate_opencv.py -m eyes 
```

### Hardware
* [SolidRun Hailo-15H-EVK](https://www.solid-run.com/hailo-15-som/)


# Special thanks
* [Shimon Faitelson (project curator)](https://il.linkedin.com/in/shimon-faitelson-22975813)
* 
