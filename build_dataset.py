import os
import csv
import cv2
import numpy as np

from face_tracker import FaceTracker
from feature_extractor import FeatureExtractor

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "expressions.csv")

def make_sure_data_dir_exists():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def file_exists(path):
    return os.path.isfile(path)

def write_header_if_needed(num_features):
    # If file already exists, do nothing
    if file_exists(CSV_PATH):
        return

    header = ["label"]
    for i in range(num_features):
        header.append("f_" + str(i))

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

def main():
    make_sure_data_dir_exists()

    print("facial data collector 3000")
    label = input("Enter label for this session: ").strip()
    if label == "":
        print("No label given. Exiting.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    tracker = FaceTracker()
    extractor = FeatureExtractor()

    print("")
    print("Instructions:")
    print("  - Show your face clearly to the camera.")
    print("  - Press 'r' to START/STOP recording frames for label:", label)
    print("  - Press 'q' to quit.")
    print("")

    recording = False
    saved_count = 0
    first_feature_len = None
    MAX_SAMPLES = 400

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read frame from webcam.")
            break

        # detect face and draw mesh
        landmarks, annotated = tracker.detect_yo_face(frame, draw=True)

        # if we are recording and we have landmarks, extract and save features
        if recording and landmarks is not None:
            features = extractor.extract(landmarks)

            if features is not None:
                # figure out header on first valid feature vector
                if first_feature_len is None:
                    first_feature_len = len(features)
                    write_header_if_needed(first_feature_len)

                row = [label] + features.tolist()

                # append row
                with open(CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                saved_count += 1
                
                # Stop after reaching max samples
                if saved_count >= MAX_SAMPLES:
                    print("Reached maximum of {} samples. Stopping.".format(MAX_SAMPLES))
                    break

        # draw status text on the frame
        status_text = "REC" if recording else "IDLE"
        color = (0, 0, 255) if recording else (255, 255, 255)
        cv2.putText(
            annotated,
            "Label: {} | Status: {} | Saved: {}/{}".format(label, status_text, saved_count, MAX_SAMPLES),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )

        cv2.imshow("PNG-VTuber Dataset Builder", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            print("Recording:", recording)

    cap.release()
    cv2.destroyAllWindows()
    print("Done. Total frames saved for label '{}': {}".format(label, saved_count))
    print("Saved to:", CSV_PATH)

if __name__ == "__main__":
    main()
