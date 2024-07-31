from firebase_utils.db import db


import datetime
import threading
from queue import Queue


class UploadThread(threading.Thread):

    def __init__(self, event, data_queue):
        super(UploadThread, self).__init__()
        self.stop_flag = False
        self.event = event
        self.data_queue = data_queue

    def run(self):
        while not self.stop_flag:
            self.event.wait()  # Block until the event is set
            if not self.data_queue.empty():
                in_count, out_count = self.data_queue.get()
                self._upload_counts(in_count, out_count)
                # print(f"Data uploaded")
            self.event.clear()  # Reset the event

    def stop(self):
        self.stop_flag = True

    def _upload_counts(self, in_count: int, out_count: int):
        new_record_ref = db.collection("counts").document()
        data = {
            "in_count": in_count,
            "out_count": out_count,
            "timestamp": datetime.datetime.now(tz=datetime.timezone.utc),
        }
        new_record_ref.set(data)
