from pathlib import Path
from PIL import Image
from uuid import uuid4
import numpy as np
import cv2
from time import time
from typing import Tuple

def classify_with_yolo(image, model):
    current_dir = Path(__file__).parent
    detection_id = uuid4()
    output_path = current_dir / "predictions" / f"yolo_{detection_id}.jpg"
    result = model(image)[0]
    # https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.plot
    im_array = result.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.save(output_path)  # save image
    return output_path


def classify_with_vgg(image, model) -> Tuple[bytes, str]:
    start_time = time()
    current_dir = Path(__file__).parent
    detection_id = uuid4()
    output_path = current_dir / "predictions" / f"vgg_{detection_id}.jpg"

    # 预处理图像
    resized_image = cv2.resize(image, (224, 224))  # 调整大小为模型的输入尺寸
    normalized_image = resized_image / 255.0  # 归一化像素值 # type: ignore

    predictions = model.predict(np.expand_dims(normalized_image, axis=0))  # type: ignore

    class_names = ["cat", "dog"]  # 替换为你的类别名称
    top_class_index = np.argmax(predictions)
    top_class = class_names[top_class_index]
    confidence = predictions[0][top_class_index]

    # # 绘制带有颜色框的窗口
    annotated_image = image.copy()

    elaspes_time = round((time() - start_time) * 1000)
    # 在图像上添加文本
    text = f"{top_class} ({confidence:.2f}) in {elaspes_time}ms"
    text_color = (255, 255, 255)  # 文本颜色，这里使用白色
    text_position = (10, 30)  # 文本位置（假设左上角为起点）
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    annotated_image = cv2.putText(
        annotated_image,
        text,
        text_position,
        font,
        font_scale,
        text_color,
        font_thickness,
    )
    ret, buffer = cv2.imencode(".jpg", annotated_image)

    return (buffer.tobytes(), top_class)
