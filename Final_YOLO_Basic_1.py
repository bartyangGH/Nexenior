import cv2
import numpy as np

#cfg = "/Users/hy/Documents/M_Project/YOLO/yolov3.cfg"
#weights = "/Users/hy/Documents/M_Project/YOLO/yolov3.weights"

# 載入YOLO的權重與模型結構
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# 獲取所有的層名稱
layer_names = net.getLayerNames()


# 定義 output_layers
output_layer_indexes = net.getUnconnectedOutLayers().flatten()
output_layers = [layer_names[i - 1] for i in output_layer_indexes]



# 可以使用coco.names檔案裡的名稱，這裡只列部分為範例
classes = ['person', 'some other class', 'some other class', '...']

# 抓取RTSP串流
cap = cv2.VideoCapture('rtsp://192.168.0.68:8554/mjpeg/1')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # 顯示資訊在畫面上
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # 物件座標
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 繪製方形框框
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                label = str(classes[class_id])
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow('YOLO RTSP Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
