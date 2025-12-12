# cs-4337-FinalProject-d-naGit
Final Project for TXST CS 4337

# Object Tracking Demo (Template Matching vs OpenCV Trackers)

This project compares simple template matching with OpenCV's built-in trackers
(CSRT, KCF, MOSSE) on a user-selected object in a video.

This project uses Python 3.8+.

## Installation
Clone the repository
```bash
git clone https://github.com/d-naGit/cs-4337-FinalProject-d-naGit.git
cd cs-4337-FinalProject-d-naGit
```

## Setup
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```
If your system requires it you may also install:
```bash
pip install opencv-python
```

## Running the Object Tracker
Run the tracker (Make sure the video is visible to the repo or use the direct path on local device).
```bash 
python tracker_app.py --video "C:\path.....!!!"
```
To use a specific tracker use...
```bash
python tracker_app.py --video myvideo.mp4 --methods template csrt
```
Available methods are: template, csrt, kcf, and mosse.

