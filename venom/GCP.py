import os
import time
from typing import Optional
from uuid import uuid4

import cv2
import pymysql
import sqlalchemy
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.sql.connector import Connector, IPTypes

from utils import format_yolo_label_from_obb

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_service_key.json"
instance_connection_name = os.getenv("INSTANCE_CONNECTION_NAME")
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")
db_name = os.getenv("DB_NAME")

storage_client = storage.Client()


def upload_to_gcp_storage(frame, bucket_name: str):
    """
    將辨識完的frame上傳到Google Cloud Storage

    Args:
    frame (numpy.ndarray): 從OpenCV捕獲的圖像數據
    bucket_name (str): Google Cloud Storage中的存儲桶名稱
    """
    # 将图像编码为 JPEG 格式
    # 第一个参数是文件扩展名，表示希望编码成的格式
    # 第二个参数是图像对象
    retval, buffer = cv2.imencode(".jpg", frame)

    # 检查是否成功编码
    if retval:
        # 将编码后的图像转换成字节序列
        frame_bytes = buffer.tobytes()
    else:
        raise Exception("图像编码失败")

    # 初始化Google Cloud Storage客戶端
    bucket = storage_client.get_bucket(bucket_name)
    # 將編碼後的圖像上傳到Google Cloud Storage
    unique_file_name = str(uuid4()) + ".jpg"
    blob = bucket.blob(unique_file_name)
    blob.upload_from_string(frame_bytes, content_type="image/jpeg")
    print("Photo uploaded to GCP")

    return f"gs://{bucket_name}/{unique_file_name}"


def upload_to_cloud_sql(
    model_version: int,
    grading_level: str,
    photo_reference: str,
    label: Optional[str] = None,
):
    table_name = "grading"

    # The data we want to upload
    data = {
        "model_version": model_version,
        "grading_level": grading_level,
        "photo_reference": photo_reference,
        "label": label,
    }

    # Building an SQL INSERT statement
    insert_statement = sqlalchemy.text(
        f"""
        INSERT INTO {table_name} (model_version, grading_level, label, photo_reference)
        VALUES (:model_version, :grading_level, :label, :photo_reference)
    """
    )
    engine = connect_with_connector()

    # Using a `with` statement to ensure the connection is properly managed
    with engine.connect() as conn:
        # Execute the SQL statement
        conn.execute(insert_statement, data)
        # Commit the transaction
        conn.commit()


# Ref: https://cloud.google.com/sql/docs/mysql/connect-connectors?hl=zh-cn#python
# pip install "cloud-sql-python-connector[pymysql]"
# Local connection:
# ./cloud-sql-proxy object-detection-417617:us-central1:mysql-agent --unix-socket=/Users/poju/Documents/spider_net/venom/cloudsql --credentials-file ./gcp_service_key.json
def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of MySQL.

    Uses the Cloud SQL Python Connector package.
    """
    # Note: Saving credentials in environment variables is convenient, but not
    # secure - consider a more secure solution such as
    # Cloud Secret Manager (https://cloud.google.com/secret-manager) to help
    # keep secrets safe.

    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

    connector = Connector(ip_type)

    def getconn() -> pymysql.connections.Connection:
        conn: pymysql.connections.Connection = connector.connect(
            instance_connection_name,  # type: ignore
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
    )
    return pool


def upload_image_and_update_db(
    frame,
    bucket_name: str,
    model_version: int,
    grading_level: str,
    obb_object=None,
):
    start_time = time.time()
    output_img_url = upload_to_gcp_storage(frame, bucket_name)
    # format obb result to text for next training
    if obb_object:
        label_text = format_yolo_label_from_obb(obb_object)
        upload_to_cloud_sql(model_version, grading_level, output_img_url, label_text)
    else:
        upload_to_cloud_sql(model_version, grading_level, output_img_url)

    print(f"The function took {time.time()-start_time:.3f} seconds to complete.")


if __name__ == "__main__":
    """
    This script is used to upload dataset to db and storage."""
    from tqdm import tqdm

    bucket_name = "camvo_training_photo"
    data_to_insert = []
    for file_path in [
        "/Users/poju/Downloads/dserxc/test",
        "/Users/poju/Downloads/dserxc/valid",
        "/Users/poju/Downloads/dserxc/train",
    ]:
        for file_name in tqdm(os.listdir(file_path + "/images")):
            bucket = storage_client.get_bucket(bucket_name)
            # 將編碼後的圖像上傳到Google Cloud Storage
            unique_file_name = str(uuid4()) + ".jpg"
            blob = bucket.blob(unique_file_name)
            blob.upload_from_filename(file_path + "/images/" + file_name)
            photo_ref = f"gs://{bucket_name}/{unique_file_name}"
            table_name = "training"
            label_file_name = ".".join(file_name.split(".")[:-1]) + ".txt"
            with open(file_path + "/labels/" + label_file_name, "r") as f:
                label = f.read()

            # The data we want to upload
            data_to_insert.append(
                {
                    "photo_reference": photo_ref,
                    "label": label,
                }
            )

    # Building an SQL INSERT statement
    insert_statement = sqlalchemy.text(
        f"""
        INSERT INTO {table_name} (photo_reference, label)
        VALUES (:photo_reference, :label)
    """
    )
    engine = connect_with_connector()

    # Using a `with` statement to ensure the connection is properly managed
    with engine.connect() as conn:
        # Execute the SQL statement
        conn.execute(insert_statement, data_to_insert)
        # Commit the transaction
        conn.commit()
