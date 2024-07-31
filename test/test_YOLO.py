# Importing necessary libraries
from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
import argparse
from supervision.geometry.core import Point
import time
import nanocamera as nano
import torch


# Loading the YOLO model with pre-trained weights
model = YOLO("./pre_trained/yolov8n.pt")


# Function to parse command line arguments for the script
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    # Adding argument for setting webcam resolution
    parser.add_argument("--camera-resolution", default=[1920, 1080], nargs=2, type=int)
    parser.add_argument("--output-resolution", default=[1920, 1080], nargs=2, type=int)
    parser.add_argument("--output-fps", default=10, type=int)
    parser.add_argument("--camera-index", default=1, type=int)
    parser.add_argument(
        "--hide-camera", action="store_true", help="Hide the camera view."
    )
    parser.add_argument(
        "--isJetson",
        action="store_true",
        help="Indicate if the script is running on a Jetson device",
    )
    args = parser.parse_args()
    return args


# Parsing arguments and setting the webcam resolution
args = parse_arguments()
WIDTH, HEIGHT = args.camera_resolution
OUT_WIDTH, OUT_HEIGHT = args.output_resolution
OUT_FPS = args.output_fps

if args.isJetson:
    cap = nano.Camera(flip=0, width=WIDTH, height=HEIGHT, fps=30)
else:
    # Initializing video capture from the webcam
    cap = cv2.VideoCapture(args.camera_index)
    # Setting the resolution of the video capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Check if CUDA is available and set the device to GPU
if torch.cuda.is_available():
    print("using cuda")
    device = torch.device("cuda")
else:
    print("cuda is unavailable, using cpu")
    device = torch.device("cpu")
model.to(device)


box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

# Main loop for processing video frames
start_time = time.time()
prev_frame_time = 0
new_frame_time = 0
while True:
    # Capture frame-by-frame from the webcam
    if type(cap) is nano.NanoCam.Camera:  # type: ignore
        frame = cap.read()
    else:
        ret, frame = cap.read()  # type: ignore
        if not ret:
            break

    result = model(frame, verbose=False, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"  # type: ignore
        for _, _, confidence, class_id, _ in detections
    ]
    # Annotating the frame with bounding boxes and labels
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)  # type: ignore
    # Reigon calculate fps
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # Converting the fps into integer
    fps = f"{fps:.2f}"
    cv2.setWindowTitle("Frame", f"FPS: {fps}")
    # End region calculate fps

    if not args.hide_camera:
        # Displaying the frame with annotated detections
        cv2.imshow("Frame", frame)

    # video_writer.write(frame)
    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Releasing the video capture and closing all OpenCV windows
cap.release()
# video_writer.release()  # Release the video writer
cv2.destroyAllWindows()
# if streamer is not None and streamer.stdin is not None:
# streamer.stdin.close()
# streamer.wait()
