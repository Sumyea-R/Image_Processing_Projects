import os
from db_setup import SessionLocal, TrainingImage

LABELS = ["Capacitor", "IC", "Resistor"]
BASE_DIR = "static/train"

db = SessionLocal()

for label in LABELS:
    folder = os.path.join(BASE_DIR, label)
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = TrainingImage(filename=filename, filepath=filepath, label=label)
        db.add(img)

db.commit()
db.close()
