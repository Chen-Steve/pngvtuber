import os
import cv2
import numpy as np

from face_tracker import FaceTracker
from feature_extractor import FeatureExtractor
from action_classifier import ActionClassifier

MONKEY_CRIB = "monkeyfaces"
EMOTIONS = ["happy", "sad", "excited", "angry", "shocked"]

def load_monkey_images():
    monkey_images = {}

    for emo in EMOTIONS:
        filename = emo + ".png"
        path = os.path.join(MONKEY_CRIB, filename)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Warning: could not load monkey image:", path)
            continue

        # If PNG has alpha channel (BGRA), convert to BGR
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        monkey_images[emo] = img

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

        # detect face + landmarks and draw mesh on a copy of the frame
        landmarks, annotated = tracker.detect_yo_face(frame, draw=True)

        emotion = None

        # extract features and predict emotion if we have a face
        if landmarks is not None:
            features = extractor.extract(landmarks)
            if features is not None:
                emotion = classifier.predict(features)

        # draw emotion text on webcam side
        if emotion is not None:
            cv2.putText(
                annotated,
                "Emotion: {}".format(emotion),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        display_frame = annotated

        # show monkey on the right if we have a matching PNG
        if emotion is not None and emotion in monkey_images:
            monkey = monkey_images[emotion]

            # resize monkey image to match webcam frame height
            h, w, c = annotated.shape
            mh, mw, mc = monkey.shape

            scale = h / float(mh)
            new_width = int(mw * scale)
            monkey_resized = cv2.resize(monkey, (new_width, h))

            # stack side by side: [webcam | monkey]
            display_frame = np.hstack([annotated, monkey_resized])

        cv2.imshow("PNG-VTuber", display_frame)

        # q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
