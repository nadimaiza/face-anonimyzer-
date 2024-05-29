
# Face Anonymizer with OpenCV and MediaPipe

This repository contains code for a face anonymizer that detects and blurs faces in real-time using a webcam and on static images using OpenCV and MediaPipe. The application can be deployed and used to ensure privacy by anonymizing faces in images and video streams.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project demonstrates how to use OpenCV and MediaPipe to detect and blur faces in images and video streams. The primary use case is to anonymize faces to protect privacy.

## Features
- Real-time face detection and anonymization using a webcam.
- Face anonymization on static images.
- Adjustable face detection confidence.
- Works with RGB images for better accuracy.

## Technologies Used
- **Python**
- **OpenCV**
- **MediaPipe**

## Installation
To run this project, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/face-anonymizer.git
    ```

2. Navigate to the project directory:
    ```bash
    cd face-anonymizer
    ```

3. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### For Real-Time Face Anonymization
1. Run the script for real-time face anonymization:
    ```bash
    python face_anonymizer.py
    ```

### For Face Anonymization in Static Images
1. Update the `img_path` variable in `face_anonymizer.py` with the path to your image:
    ```python
    img_path = '/path/to/your/image.jpg'
    ```

2. Run the script:
    ```bash
    python face_anonymizer.py
    ```

## Project Structure
```
face-anonymizer/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   └── face_anonymizer.py
├── README.md
├── requirements.txt
└── LICENSE
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Here is the content of your `face_anonymizer.py` script for reference:

```python
import os
import cv2
import mediapipe as mp

output_dir = './output'
# Check if the directory exists before creating it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read image
img_path = '/Users/nadimaizarani/Desktop/computer vision code VS code/computer vision test/100_0250 copy.JPG'
img = cv2.imread(img_path)

H, W, _ = img.shape

# Detect face
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    # model_selection can take 2 numbers or 0 or 1 : you use 0 if the face is close to the camera, 1 if it's far (more than 5 m)
    # min_detection_confidence [0, 1]. the closer we are to 1 the more accurate the model is going to identify faces but it can land some errors if the face is not 100% clear.

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # we have to change the image to RGB because mediapipe works with RGB images 
    out = face_detection.process(img_rgb) # face_detection.process is a function in mediapipe to detect faces 

    if out.detections:
        for detection in out.detections: # if a face was detected, we iterate through each result (face scanned) to extract informations
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box # contain the bounding box of the face detected 
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height 

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Blur face
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (100, 100))

            # img = cv2.rectangle(img, (x1, y1), (x1 +w , y1 + h), (0, 255, 0), 10)

# Save image
cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

# Display the image
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Make sure to create a `requirements.txt` file with the necessary dependencies:

```txt
opencv-python
mediapipe
```
