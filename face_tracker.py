import cv2
import mediapipe as mp
import numpy as np

class FaceTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, # optimized for video
            max_num_faces=1, # just one face
            min_detection_confidence=0.5 # just 50% confidence to detect a face
        )
    
    def detect_yo_face(self, frame):
        # convert bgr to rgb (mediapipe needs rgb)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0] # get first face
            frame_height = frame.shape[0] # height in pixels
            frame_width = frame.shape[1] # width in pixels
            
            # convert normalized coordinates (0.0-1.0) to pixel positions
            landmarks = []
            for landmark in face_landmarks.landmark:
                x_pixel = landmark.x * frame_width
                y_pixel = landmark.y * frame_height
                z_pixel = landmark.z * frame_width
                landmarks.append([x_pixel, y_pixel, z_pixel])
            
            return np.array(landmarks), None
        
        return None, None
