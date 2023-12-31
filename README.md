# Stress Detection Project Documentation

## Project Overview

This project aims to analyze human behavior in videos to detect stress levels based on various facial and body features. The stress detection is performed by analyzing factors such as blink frequency, eyebrow movement, emotions expressed, lip movements, hand movements, gaze direction, and face orientation.

## Dependencies

- OpenCV: 4.5.3
- Mediapipe: 0.8.7
- Matplotlib: 3.4.3
- Math (Standard Python Library)

## Setup Instructions

1. Clone the repository: `git clone https://github.com/your-username/your-repository.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python stress_detection.py --video_path "path/to/your/video.mp4"`

## Usage

The script analyzes stress factors in a given video and generates a graph depicting stress levels over time. You can customize the input video path using the `--video_path` argument.

```bash
python stress_detection.py --video_path "path/to/your/video.mp4"
Feature Explanation
Blink Detection
The number of blinks per second is calculated to detect stress. Increased blink rates might indicate stress.

Eyebrow Movement
The angle between eyebrow points is analyzed to understand eyebrow movements, which can contribute to stress detection.

Emotions
Complex emotions analysis logic is applied based on eye distance and eyebrow distance.

Lip Movements
Lip distance is calculated to assess stress levels related to facial expressions and lip movements.

Hand Movement
The total Euclidean distance between consecutive hand landmarks is measured to detect stress from hand movements.

Gaze Direction
The ratio of horizontal to vertical eye movement is calculated to understand gaze direction, contributing to stress detection.

Face Orientation
The roll angle of the face, representing head tilting sideways, is measured for stress analysis.

Customization
You can customize the weights of each feature in the calculate_stress function in the script.

Results and Output
The script generates a stress level graph over time, providing insights into the individual factors contributing to stress detection.

Contact Information
For questions or feedback, contact: Eljaziamal@gmail.com
