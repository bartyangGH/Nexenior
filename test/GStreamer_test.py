import cv2
import numpy as np
import time

kvs_out_stream_name = "ExampleStream"
access_key = "AKIAT4HA3AUEDB5UV44S"
secret_key = "ElHqPOQRf/cFGJ3Eq/gdRcMEM6ddjKj7o+RRhbDD"
region = "us-east-1"

pipeline = f"appsrc ! videoconvert ! video/x-raw,format=BGR,width={400},height={400},framerate=25/1 ! videoconvert ! video/x-raw ! x264enc key-int-max=45 ! video/x-h264,stream-format=avc,alignment=au,profile=baseline ! kvssink stream-name=ExampleStream storage-size=512 aws-region={region} access-key={access_key} secret-key={secret_key}"

out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, float(25), (400, 400), True)
if not out.isOpened():
    print("无法打开视频流或文件")
    exit()

while True:
    frame = (np.random.rand(400, 400, 3) * 255).astype(np.uint8)
    out.write(frame)
    time.sleep(0.04)
