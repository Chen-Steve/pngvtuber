import os
import cv2
import numpy as np

from face_tracker import FaceTracker
from feature_extractor import FeatureExtractor
from action_classifier import ActionClassifier

MONKEY_DIR = "monkeyfaces"
EMOTIONS = ["happy", "sad", "excited", "angry", "shocked"]

def load_monkey_images():
    monkey_images = {}

    exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

    for emo in EMOTIONS:
        for ext in exts:
            path = os.path.join(MONKEY_DIR, emo + ext)
            if os.path.isfile(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                # convert BGRA -> BGR if PNG has alpha
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                monkey_images[emo] = img
                break

    return monkey_images

def main():
    cap = cv2.VideoCapture(0)
    tracker = FaceTracker()
    extractor = FeatureExtractor()
    classifier = ActionClassifier("expression_model.pkl")

    monkey_images = load_monkey_images()

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        landmarks, _ = tracker.detect_yo_face(frame, draw=False)

        emotion = None

        if landmarks is not None:
            features = extractor.extract(landmarks)
            if features is not None:
                emotion = classifier.predict(features)

        # draw emotion text
        annotated = frame.copy()
        if emotion is not None:
            cv2.putText(
                annotated,
                "Emotion: {}".format(emotion),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        display_frame = annotated

        # show matching monkey on the right
        if emotion is not None and emotion in monkey_images:
            monkey = monkey_images[emotion]

            h, w, _ = annotated.shape
            mh, mw, _ = monkey.shape

            scale = h / float(mh)
            new_width = int(mw * scale)
            monkey_resized = cv2.resize(monkey, (new_width, h))

            display_frame = np.hstack([annotated, monkey_resized])

        cv2.imshow("PNG-VTuber", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
