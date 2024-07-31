import threading

from ultralytics import YOLO
import cv2
from GCP import upload_image_and_update_db

# 加载模型
tablet_detection_model = YOLO("tablet_detect_n.pt", task="detect")
# tablet_seg_model = YOLO("tablet_seg_m.pt")
model_name = "crack_detection_threeCat_800_crop_obb_m_20ep"
crack_detection_model = YOLO(f"{model_name}.pt", task="obb")
MODEL_VERSION_MAP = {
    "crack_detection_threeCat_800_crop_obb_m_20ep": 1,
    "crack_detection_threeCat_800_crop_obb_n_20ep": 2,
    "crack_detection_threeCat_500_aug_obb_n": 3,
    "crack_detection_twoCat_obb_n": 4,
    "test_workflow": 5,
}
model_version = MODEL_VERSION_MAP[model_name]
"""
good:
id:1 crack_detection_threeCat_800_crop_obb_m_20ep.pt (low performance)
id:2 crack_detection_threeCat_800_crop_obb_n_20ep.pt (low performance)
id:3 crack_detection_threeCat_500_aug_obb_n (100eps, low val performance)
id:4 crack_detection_twoCat_obb_n (150eps early ends, low val performance)
"""

# 打开视频文件
video_path = "ipad_test.mp4"
image_path = "cracked2.jpeg"
cap = cv2.VideoCapture(video_path)
#3 检查是否成功打开视频文件
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
# 获取原视频的帧率和帧大小以用于输出视频
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# 定义输出视频的编码和名称
# output_path = "annotated_output_video.mp4"
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 或者使用 'XVID' 如果MP4不工作
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

detection_ongoing = {}
detection_completed = {}  # potential memory leak
TABLET_THRESHOLD = 0.8
CRACK_THRESHOLD = 0.5
# if we didn't detect any of cracks in this number of frames, than classify
# the tablet as 'OK'
CLASSIFY_THRESHOLD = 18
# Start detect after this number of frames detected for an object
START_DETECT_THRESHOLD = 15
# warm up crack detection model
warmup = False
# 逐帧读取视频
while True:
    # 读取下一帧
    ret, frame = cap.read()

    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if not warmup:
        crack_detection_model.predict(source=frame)
        warmup = True

    # 将帧传递给平板检测模型
    result = tablet_detection_model.track(
        source=frame, persist=True, conf=TABLET_THRESHOLD, verbose=False
    )[0]
    annotated = result.plot()

    boxes = result.boxes  # Boxes对象，用于边界框输出
    if boxes and boxes.id:
        for i, box in enumerate(boxes):
            tracker_id = int(boxes.id[i].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if detection_ongoing.get(tracker_id, 0) < START_DETECT_THRESHOLD:
                if tracker_id in detection_completed:
                    cv2.putText(
                        annotated,
                        detection_completed[tracker_id],
                        (x1, y1 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (36, 255, 12),
                        3,
                    )
                    continue
                else:
                    detection_ongoing[tracker_id] = (
                        detection_ongoing.get(tracker_id, 0) + 1
                    )
                    continue

            crop_img = frame[y1:y2, x1:x2]
            # 将裁剪的图像传递给裂缝检测模型
            crack_detection_result = crack_detection_model.predict(
                source=crop_img, conf=CRACK_THRESHOLD
            )[0]
            if crack_detection_result.obb:
                detection_completed[tracker_id] = "Cracked"
                del detection_ongoing[tracker_id]
                upload_thread = threading.Thread(
                    target=upload_image_and_update_db,
                    args=(
                        crop_img,
                        "camvo-model-output",
                        model_version,
                        "Cracked",
                        crack_detection_result.obb,
                    ),
                )
                upload_thread.start()

            else:
                if detection_ongoing.get(tracker_id, 0) < CLASSIFY_THRESHOLD:
                    detection_ongoing[tracker_id] = (
                        detection_ongoing.get(tracker_id, 0) + 1
                    )
                else:
                    detection_completed[tracker_id] = "OK"
                    del detection_ongoing[tracker_id]
                    upload_thread = threading.Thread(
                        target=upload_image_and_update_db,
                        args=(
                            crop_img,
                            "camvo-model-output",
                            model_version,
                            "OK",
                        ),
                    )
                    upload_thread.start()
    # out.write(frame)
    # 显示带有标注的帧
    cv2.imshow("Frame", annotated)

    # 按'q'退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放VideoCapture对象
cap.release()
# out.release()
cv2.destroyAllWindows()
