import argparse
import cv2

# Create the parser
parser = argparse.ArgumentParser(description="Webcam stream index")

# Add the arguments
parser.add_argument(
    "WebcamIndex",
    metavar="webcam_index",
    type=int,
    help="The index of the webcam stream",
)
parser.add_argument(
    "--isJetson",
    action="store_true",
    help="Indicate if the script is running on a Jetson device",
)


# Parse the arguments
args = parser.parse_args()

if args.isJetson:
    cap = cv2.VideoCapture(
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)29/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink",
        cv2.CAP_GSTREAMER,
    )
else:
    # Create a VideoCapture object
    cap = cv2.VideoCapture(args.WebcamIndex)

# Check if the stream is opened correctly
if not cap.isOpened():
    print("Error: Could not open stream")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow("Frame", frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # When everything done, release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()
