import cv2
import mediapipe as mp
import numpy as np

class FaceTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,      # optimized for video
            max_num_faces=1,              # just one face
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    
    def detect_yo_face(self, frame, draw=False):
        # BGR -> RGB (mediapipe wants RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        landmarks = None
        annotated = frame.copy()

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # convert normalized coords (0–1) to pixel positions
            pts = []
            for lm in face_landmarks.landmark:
                x_pixel = lm.x * w
                y_pixel = lm.y * h
                z_pixel = lm.z * w
                pts.append([x_pixel, y_pixel, z_pixel])
            landmarks = np.array(pts)

            if draw:
                self.mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                )

        return landmarks, annotated
