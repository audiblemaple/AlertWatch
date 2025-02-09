# AlertWatch
<img src="Capstone_Project_Docs/Banner.png" alt="AlertWatch Banner">

## Abstract
In modern environments, whether in vehicles, offices, or other settings, maintaining alertness and preventing fatigue-related incidents has become a critical concern. "AlertWatch" presents an innovative solution utilizing advanced computer vision and hardware to monitor individuals and detect signs of drowsiness in real-time. This project leverages facial recognition and facial landmark detection to continuously assess the alertness levels of a person. Upon detecting signs of drowsiness, "AlertWatch" triggers immediate alerts to prevent potential accidents, lapses in productivity or even security concerns. This paper details the development process of "AlertWatch," focusing on its implementation of real-time monitoring through cameras and software algorithms, as well as outlining the development process and hardware options. "AlertWatch" enhances safety and efficiency and sets new standards for proactive alertness management through technology.

## Project Docs
* [Project Book part 1](Capstone_Project_Docs/Part_1/AlertWatch_phase_A.docx)
* [Project Presentation part 1](Capstone_Project_Docs/Part_1/AlertWatch_phase_A.pptx)
---
* [Project Book part 2](Capstone_Project_Docs/Part_2/AlertWatch_phase_B.docx)
* [Project Poster part 2](Capstone_Project_Docs/Part_2/AlertWatch_phase_B.pdf) (give it time to load, it's a big file)
* [Project Video part 2](Capstone_Project_Docs/Part_2/AlertWatch_phase_B.mp4)


## Developer documentation 
[DEV docs](https://audiblemaple.github.io/AlertWatch/)

## Components
The system comprises 2 parts:
* [ECU / Management unit](#ecu--management-unit)
* [Detection unit](#detection-unit)
<br>

## ECU / Management unit
### Software
* [ECU web UI](ECU/server/frontend)
* [ECU backend](ECU/server)

### Hardware
* [SolidRun Bedrock R7000](https://www.solid-run.com/industrial-computers/bedrock-r7000-edgeai/)
<br>

## Detection unit
### Software
* [Detection unit code](Production/detector)

### Hardware
* [SolidRun Hailo-15H-EVK](https://www.solid-run.com/hailo-15-som/)
<br>

## Models
### Hailo 8
* [Face detection](Production/models)
* [Face landmark detection](Production/models)

### Hailo 15H
* [Face detection](Production/models)
* [Face landmark detection](Production/models)

### Speech recognition
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)

<br>

# Special thanks
* Shimon Faitelson (project curator): [LinkedIn ](https://il.linkedin.com/in/shimon-faitelson-22975813)
* Mikhail Anikin (professional consultation): [GitHub](https://github.com/anikinmd) | [LinkedIn ](https://www.linkedin.com/in/mikhail-anikin/)
* My friends for pushing me and keeping me sane during this degree
  * Gilad Segal:   [GitHub](https://github.com/gilseg10) | [LinkedIn ](https://www.linkedin.com/in/gilad-segal-4aa158267/)
  * Ron Shahar:    [GitHub](https://github.com/Seth7171) | [LinkedIn ](https://www.linkedin.com/in/ron-shahar7171/)
  * Nitzan maman:  [GitHub](https://github.com/NitsanMaman) | [LinkedIn ](https://www.linkedin.com/in/nitsan-maman-a6b35a139/)
  * Elad Fischer:  [GitHub](https://github.com/EladFis03) | [LinkedIn ](https://www.linkedin.com/in/elad-fisher-064101198/)
  * Amir Mishayev: [Github](https://github.com/amir3x0) | [LinkedIn](https://www.linkedin.com/in/amir-mishayev/)
