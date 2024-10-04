# Facial landmarks detection

## Introduction
The project is based on Machine learning models to detect users faces, crop and detect 68 facial landmarks.\
Trying to assess the users' alertness.
If a drop in alertness is detected, the detection system will signal the ECU to alert the user.

## Project Docs
* [Book](Docs/AlertWatch.docx)
* [Presentation](Docs/Alertwatch.pptx)

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