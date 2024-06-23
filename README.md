# Assisting Visually Impaired Individuals
### University Project for AOOP @ [IPVC-ESTG](https://www.ipvc.pt/estg/)

#### Problem: Helping visually impaired individuals navigate safely

#### Description: We will Utilize YOLO to detect obstacles and important objects to assist visually impaired individuals in navigating their environment.

## High Level Overview
![high_level_overview](https://github.com/carlosmacho/AOOP-CP3/assets/61155153/71c66383-5273-4247-9902-ca1303417aa3)

#### Implementation Steps:

1. **Data Collection:** Gather images of common obstacles and important objects (e.g., doors, stairs, etc.).
2. **Model Training:** Train a YOLO model to detect these obstacles and objects.
3. **Real-Time Detection:** Use a portable camera system to provide real-time detection.
4. **Guidance System:** Develop an audio feedback system to inform users about detected objects and obstacles.

#### Tools:

- **YOLOv8:** For real-time object detection.
- **OpenCV:** For video capture and processing.
- **gTTS:** For text-to-speech conversion to provide audio feedback.
- **Python:** For scripting and integrating the components.
