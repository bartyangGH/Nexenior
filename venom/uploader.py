from firebase_utils.app import bucket
import os


def upload_to_firebase(file_path):
    file_name = os.path.basename(file_path)
    blob = bucket.blob("venom/" + file_name)
    blob.upload_from_filename(file_path)
