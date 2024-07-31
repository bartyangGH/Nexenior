import firebase_admin
from firebase_admin import credentials, firestore, storage
import os

# Get the absolute path to the directory of the current file
dir_path = os.path.dirname(os.path.realpath(__file__))

# Create the absolute path to credentials.json
cred_path = os.path.join(dir_path, "../credentials.json")
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

db = firestore.client()
Bucket = storage.bucket()

if __name__ == "__main__":
    doc_ref = db.collection("users").document("alovelace")
    doc_ref.set({"first": "Ada", "last": "Lovelace", "born": 1815})
