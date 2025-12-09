import numpy as np

class FeatureExtractor:
    def __init__(self):
        # empirically good anchor points in FaceMesh indices
        self.NOSE_INDEX = 1          # near nose bridge
        self.LEFT_EYE_OUTER = 33     # outer corner of left eye
        self.RIGHT_EYE_OUTER = 263   # outer corner of right eye

    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        if landmarks is None:
            return None

        # 1) center at nose so position in frame doesn’t matter
        nose = landmarks[self.NOSE_INDEX]  # (3,)
        centered = landmarks - nose        # (468, 3)

        # 2) normalize scale using eye distance (roughly constant for a face)
        left_eye = landmarks[self.LEFT_EYE_OUTER]
        right_eye = landmarks[self.RIGHT_EYE_OUTER]
        eye_dist = np.linalg.norm(left_eye - right_eye)
        if eye_dist < 1e-6:
            eye_dist = 1e-6  # avoid divide by zero

        normalized = centered / eye_dist   # (468, 3)

        # 3) drop z for now (x,y is usually enough) and flatten
        xy = normalized[:, :2]             # (468, 2)
        features = xy.flatten()            # (468*2,) = (936,)

        return features

    def extract_batch(self, landmarks_list):
        feats = []
        for lm in landmarks_list:
            f = self.extract(lm)
            if f is not None:
                feats.append(f)
        if not feats:
            return None
        return np.stack(feats, axis=0)
