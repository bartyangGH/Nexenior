# Importing necessary libraries
from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
import argparse
from supervision.geometry.core import Point
import time
import threading
import queue
# from UploadThread import UploadThread
import nanocamera as nano
import torch

# from Streamer import get_streamer

# Create an Event object
upload_event = threading.Event()

# Create a queue to share data between threads
data_queue = queue.Queue()

# Start a new thread
# upload_thread = UploadThread(upload_event, data_queue)
# upload_thread.start()

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

# Set up Video Writer
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
# output_video_path = "output.mp4"  # Output file name
# fps = 24  # Frame rate
# video_writer = cv2.VideoWriter(
#     output_video_path, fourcc, fps, (WIDTH, HEIGHT)
# )

# Set up Video Streamer
# streamer = get_gstreamer(WIDTH, HEIGHT)

# kvs_out_stream_name = "Spidernet"
# access_key = "AKIAT4HA3AUEDB5UV44S"
# secret_key = "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"
# region = "us-east-1"

# pipeline = f"appsrc ! videoconvert ! video/x-raw,format=BGR,width={OUT_WIDTH},height={OUT_HEIGHT},framerate={OUT_FPS}/1 ! videoconvert ! video/x-raw ! x264enc key-int-max=45 ! video/x-h264,stream-format=avc,alignment=au,profile=baseline ! kvssink stream-name={kvs_out_stream_name} storage-size=512 aws-region={region} access-key={access_key} secret-key={secret_key}"

# out = cv2.VideoWriter(
#     pipeline, cv2.CAP_GSTREAMER, 0, float(OUT_FPS), (OUT_WIDTH, OUT_HEIGHT), True
# )

byte_tracker = sv.ByteTrack()
trace_annotator = sv.TraceAnnotator()
# Defining a polygon zone for detection area
# ZONE_POLYGON = np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1]])
# zone_polygon = (ZONE_POLYGON * np.array((WIDTH, HEIGHT))).astype(int)
# Creating a zone for detection within the defined polygon
# zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=(WIDTH, HEIGHT))
# Annotator for the defined polygon zone
# zone_annotator = sv.PolygonZoneAnnotator(
#     zone=zone, color=sv.Color.red(), thickness=2, text_thickness=4, text_scale=2
# )
# Creating a line zone for detection
line = sv.LineZone(Point(WIDTH // 2, 0), Point(WIDTH // 2, HEIGHT))
line_annotator = sv.LineZoneAnnotator(text_scale=2)

# Annotator for bounding boxes around detected objects
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
    # Perform object detection on the frame using YOLOv8
    result = model(frame, verbose=False, agnostic_nms=True)[0]
    # Converting detections to supervision format
    detections = sv.Detections.from_ultralytics(result)
    # Filtering detections for "Person" class (class_id = 0)
    detections = detections[detections.class_id == 0]
    # Updating tracker with current frame detections
    detections = byte_tracker.update_with_detections(detections)
    # Preparing labels for each detection
    labels = [
        f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"  # type: ignore
        for _, _, confidence, class_id, tracker_id in detections
    ]
    # Annotating the frame with bounding boxes and labels
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)  # type: ignore
    # (The following line is commented out as it's not used)
    # zone.trigger(detections=detections)
    # Annotating the frame with object traces
    frame = trace_annotator.annotate(scene=frame, detections=detections)
    # Triggering line zone detection and annotating the frame
    line.trigger(detections=detections)
    frame = line_annotator.annotate(frame=frame, line_counter=line)

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

    # Reigon streaming
    # if streamer is not None and streamer.stdin is not None:
    #     streamer.stdin.write(frame.tobytes())
    # out.write(cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_AREA))
    # End region streaming

    # Reigon upload
    # If 5 seconds have passed, add counts to the queue
    if time.time() - start_time >= 10:
        data_queue.put((line.in_count, line.out_count))
        upload_event.set()
        start_time = time.time()
    # End reigon upload

    # video_writer.write(frame)
    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# upload_thread.stop()
upload_event.set()  # Set the event to unblock the thread
# Releasing the video capture and closing all OpenCV windows
cap.release()
# video_writer.release()  # Release the video writer
cv2.destroyAllWindows()
# if streamer is not None and streamer.stdin is not None:
# streamer.stdin.close()
# streamer.wait()
