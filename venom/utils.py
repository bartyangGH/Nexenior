def format_yolo_label_from_obb(obb_object) -> str:
    res = ""
    for i, cls in enumerate(obb_object.cls):
        nested_list = obb_object.xyxyxyxyn[i].tolist()
        flat_list = [item for sublist in nested_list for item in sublist]
        res += f"{int(cls)} {' '.join(map(str, flat_list))}\n"
    return res
