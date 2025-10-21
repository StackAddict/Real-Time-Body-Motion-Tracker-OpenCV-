🚀 Features

Real-time motion detection from webcam feed (cv2.VideoCapture).

Object tracking using contour detection and matching.

Smart classification — approximates head vs. arms based on bounding box aspect ratios and position.

Speed visualization — arrows show movement direction and magnitude.

Rapid movement detection (e.g., sudden gestures).

FPS display and debug overlay.

Automatic tracking IDs assigned to moving objects.

Robust error handling and graceful shutdown.

🧩 How it works

Captures consecutive frames from the camera.

Computes frame differences using OpenCV’s absdiff, cvtColor, and thresholding.

Extracts contours representing moving regions.

Filters contours by size and shape to reduce noise.

Matches contours frame-to-frame to maintain tracked IDs.

Calculates displacement and draws direction arrows.

Highlights:

Blue = Head-like region

Green = Arms or general motion

Red = Rapid movement (> speed threshold)

Red arrows = Movement direction

Displays live visualization with performance (FPS) info.

🧠 Tech Stack

Language: Python 3.x

Libraries:

opencv-python (cv2)

numpy

math, time

⚙️ Installation
# Clone the repository
git clone https://github.com/yourusername/MotionSentinel.git
cd MotionSentinel

# Install dependencies
pip install opencv-python numpy

▶️ Usage

Run the script directly:

python motion_sentinel.py


You’ll see:

A camera window titled “Human Body Tracking”.

Motion areas highlighted in color.

Real-time FPS and tracked object counts.

Press q to quit safely.

🔧 Key Parameters

You can tweak sensitivity and behavior inside the script:

Variable	Description	Default
min_contour_area	Minimum motion region size	900
max_match_distance	Max distance to match previous objects	120
rapid_movement_threshold	Speed threshold for red “RAPID” tag	25
arrow_scale	Length of direction arrows	35
arrow_smoothing	Frames used to average direction	15
min_aspect_ratio / max_aspect_ratio	Filters shape proportions	0.3 / 2.5
💡 Example output

When you move in front of the camera:

A green box outlines your arms/body.

A blue box marks your head.

A red box + “RAPID” label shows sudden movement.

A red arrow points in the direction of movement.

FPS counter shows top-left; total tracked objects are listed.

🧰 Advanced ideas

Save motion logs or video clips when rapid motion is detected.

Integrate object classification using a neural network for better accuracy.

Use multi-camera or CCTV feeds for real surveillance.

Add sound or alert notifications for motion triggers.

⚠️ Notes

Requires a working webcam and sufficient lighting.

May need calibration for different environments (min area, thresholds).

Designed for local monitoring and experimentation, not for production surveillance.

Works cross-platform on Windows, Linux, macOS.

🧾 License

MIT License — free to use, modify, and distribute.
