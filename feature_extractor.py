import numpy as np

class FeatureExtractor:
    def __init__(self):
        # empirically good anchor points in FaceMesh indices
        self.NOSE_INDEX = 1          # near nose bridge
        self.LEFT_EYE_OUTER = 33     # outer corner of left eye
        self.RIGHT_EYE_OUTER = 263   # outer corner of right eye
        
        # Key landmark indices for emotion-specific features found online from 
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

        # Mouth landmarks
        self.UPPER_LIP_TOP = 13
        self.LOWER_LIP_BOTTOM = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        
        # Eyebrow landmarks (inner and outer)
        self.LEFT_BROW_INNER = 107
        self.LEFT_BROW_OUTER = 70
        self.RIGHT_BROW_INNER = 336
        self.RIGHT_BROW_OUTER = 300
        
        # Eye landmarks for openness
        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374
        
        # Forehead reference (for brow distance)
        self.FOREHEAD = 10
        
        # Chin
        self.CHIN = 152

    def _compute_emotion_features(self, landmarks: np.ndarray, eye_dist: float) -> np.ndarray:
        features = []
        
        # MOUTH ASPECT RATIO (MAR) - open mouth for shocked
        mouth_top = landmarks[self.UPPER_LIP_TOP]
        mouth_bottom = landmarks[self.LOWER_LIP_BOTTOM]
        mouth_left = landmarks[self.MOUTH_LEFT]
        mouth_right = landmarks[self.MOUTH_RIGHT]
        
        mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        mar = mouth_height / (mouth_width + 1e-6)  # high = shocked/surprised
        features.append(mar)
        
        # SMILE RATIO - mouth corners relative to center
        mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2
        left_corner_y = mouth_left[1]
        right_corner_y = mouth_right[1]
        # Negative = corners up (smile), Positive = corners down (frown/sad)
        smile_left = (left_corner_y - mouth_center_y) / (eye_dist + 1e-6)
        smile_right = (right_corner_y - mouth_center_y) / (eye_dist + 1e-6)
        smile_ratio = (smile_left + smile_right) / 2
        features.append(smile_ratio)
        
        # LEFT EYEBROW HEIGHT (relative to eye) - raised = shocked, lowered = angry
        left_brow_inner = landmarks[self.LEFT_BROW_INNER]
        left_brow_outer = landmarks[self.LEFT_BROW_OUTER]
        left_eye_center = (landmarks[self.LEFT_EYE_TOP] + landmarks[self.LEFT_EYE_BOTTOM]) / 2
        left_brow_height = (left_eye_center[1] - (left_brow_inner[1] + left_brow_outer[1]) / 2) / (eye_dist + 1e-6)
        features.append(left_brow_height)
        
        # RIGHT EYEBROW HEIGHT
        right_brow_inner = landmarks[self.RIGHT_BROW_INNER]
        right_brow_outer = landmarks[self.RIGHT_BROW_OUTER]
        right_eye_center = (landmarks[self.RIGHT_EYE_TOP] + landmarks[self.RIGHT_EYE_BOTTOM]) / 2
        right_brow_height = (right_eye_center[1] - (right_brow_inner[1] + right_brow_outer[1]) / 2) / (eye_dist + 1e-6)
        features.append(right_brow_height)
        
        # BROW FURROW - distance between inner brows (close = angry)
        brow_furrow = np.linalg.norm(left_brow_inner - right_brow_inner) / (eye_dist + 1e-6)
        features.append(brow_furrow)
        
        # LEFT EYE ASPECT RATIO (EAR) - wide = shocked
        left_eye_top = landmarks[self.LEFT_EYE_TOP]
        left_eye_bottom = landmarks[self.LEFT_EYE_BOTTOM]
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        left_ear = left_eye_height / (eye_dist + 1e-6)
        features.append(left_ear)
        
        # RIGHT EYE ASPECT RATIO
        right_eye_top = landmarks[self.RIGHT_EYE_TOP]
        right_eye_bottom = landmarks[self.RIGHT_EYE_BOTTOM]
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        right_ear = right_eye_height / (eye_dist + 1e-6)
        features.append(right_ear)
        
        # JAW DROP - chin distance from nose (open jaw = shocked)
        nose = landmarks[self.NOSE_INDEX]
        chin = landmarks[self.CHIN]
        jaw_drop = np.linalg.norm(chin - nose) / (eye_dist + 1e-6)
        features.append(jaw_drop)
        
        # ASYMMETRY - difference between left/right (can indicate certain expressions)
        brow_asymmetry = abs(left_brow_height - right_brow_height)
        features.append(brow_asymmetry)
        
        # OVERALL BROW POSITION (average height)
        avg_brow_height = (left_brow_height + right_brow_height) / 2
        features.append(avg_brow_height)
        
        return np.array(features)

    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        if landmarks is None:
            return None

        # center at nose so position in frame doesn't matter
        nose = landmarks[self.NOSE_INDEX]  # (3,)
        centered = landmarks - nose        # (468, 3)

        # normalize scale using eye distance (roughly constant for a face)
        left_eye = landmarks[self.LEFT_EYE_OUTER]
        right_eye = landmarks[self.RIGHT_EYE_OUTER]
        eye_dist = np.linalg.norm(left_eye - right_eye)
        if eye_dist < 1e-6:
            eye_dist = 1e-6  # avoid divide by zero

        normalized = centered / eye_dist   # (468, 3)

        # drop z for now (x,y is usually enough) and flatten
        xy = normalized[:, :2]             # (468, 2)
        raw_features = xy.flatten()        # (468*2,) = (936,)
        
        # ADD emotion-specific engineered features
        emotion_features = self._compute_emotion_features(landmarks, eye_dist)
        
        # Combine raw landmarks with engineered features
        features = np.concatenate([raw_features, emotion_features])

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
