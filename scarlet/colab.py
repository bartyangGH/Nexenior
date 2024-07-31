from ultralytics import YOLO
import cv2
from pathlib import Path

curdir = Path(__file__).parent
file_path = curdir / "truck.mov"
model_path = curdir.parent / "pre_trained" / "yolov8n.pt"

model = YOLO(model_path)
cap = cv2.VideoCapture(str(file_path))

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Define the codec using VideoWriter_fourcc() and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640, 480))

# Read the video file frame by frame
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Display the resulting frame
        results = model(frame, verbose=False, conf=0.7)
        if len(results) > 1:
            print("More than one frame returned")
            break

        for result in results:
            im_array = result.plot()
            for box in result.boxes:
                if result.names[box.cls.item()] != "truck":
                    continue
                x, y, w, h = box.xywh.tolist()[0]  # Adjust indexing if necessary
                x, y, w, h = int(x), int(y), int(w), int(h)
                x = x - w // 2  # Convert from center to top-left
                y = y - h // 2

                # Crop the detected object from the frame
                # Ensure coordinates are within the frame dimensions
                crop_img = im_array[y : y + h, x : x + w]
                data = ocr_image(crop_img, to_data=True)
                # dict_keys(['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text'])
                print(data["text"])
                text = "".join(data["text"]).strip()
                if not text:
                    continue
                else:
                    print(text)
                n_boxes = len(data["level"])
                for i in range(n_boxes):
                    (xp, yp, wp, hp) = (
                        data["left"][i],
                        data["top"][i],
                        data["width"][i],
                        data["height"][i],
                    )
                    cv2.rectangle(
                        im_array,
                        (x + xp, y + yp),
                        (x + xp + wp, y + yp + hp),
                        (0, 255, 128),
                        2,
                    )

            out.write(im_array)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break

# When everything done, release the VideoCapture object
cap.release()
out.release()
cv2.destroyAllWindows()
