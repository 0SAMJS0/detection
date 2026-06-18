# Face & Finger Detection

A real-time webcam app that detects your face and counts how many fingers you're holding up. It draws a labeled box around your face and a box around your hand showing the live finger count.

Built with **YOLOv8** (face detection), **MediaPipe** (hand tracking), and **OpenCV** (camera + drawing).

## Features

- Detects your face with a YOLOv8 face model and labels it "My Face"
- Tracks your hand and counts raised fingers (0–5) in real time
- Runs entirely on your webcam — no internet needed after the first run

## Requirements

- Python 3.9–3.12 (MediaPipe may not install on newer versions yet)
- A webcam

## Installation

```bash
pip install ultralytics opencv-python "mediapipe==0.10.21" huggingface_hub
```

> **Note:** MediaPipe is pinned to `0.10.21` on purpose. Newer versions removed the `solutions` API this project relies on, so the latest release crashes with `module 'mediapipe' has no attribute 'solutions'`.

## Usage

```bash
python3 detect.py
```

- The first run automatically downloads a small (~6 MB) YOLOv8 face model.
- Hold your hand up to the camera to see the finger count.
- Press **q** to quit.

## How It Works

- **Face:** YOLOv8 finds faces and returns bounding boxes, which OpenCV draws on screen.
- **Fingers:** MediaPipe locates 21 landmark points on the hand. A finger counts as "up" when its tip sits higher than its knuckle; the thumb uses a distance check instead, since it points sideways.

## Troubleshooting

**macOS: "not authorized to capture video"**
Grant camera access in System Settings → Privacy & Security → Camera, enable your terminal app, then fully quit and reopen it.

**`module 'mediapipe' has no attribute 'solutions'`**
Your MediaPipe is too new. Reinstall the pinned version:

```bash
pip uninstall -y mediapipe
pip install "mediapipe==0.10.21"
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
