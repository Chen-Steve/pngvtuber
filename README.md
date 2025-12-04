# PNG VTuber CS445

### Core stuff
1. Integrate MediaPipe face detection (`face_tracker.py`)
2. Extract facial landmarks
3. Calculate features: EAR, MAR, head pose angles (`feature_extractor.py`)
4. Create character PNG database structure (`image_database.py`)
5. Implement side-by-side display: original video | character PNG (`renderer.py`)
6. Build main application loop (`main.py`)

### Training (Local + Colab)
1. Record training videos performing each gesture (10-20 samples each, 2-5 seconds per video)
2. Extract features from video frames using MediaPipe
3. Save features to CSV file
4. Upload videos to Label Studio (local)
5. Label each frame/segment with correct gesture
6. Export labels from Label Studio
7. Upload training data (features + labels) to Colab
8. Train ML model (Random Forest, SVM, idk yet) in Colab
9. Evaluate model performance (idk how yet)
10. Download trained model file (`.pkl`)

### ML Integration
1. Load trained model in `action_classifier.py`
3. Test real-time gesture recognition (eye ball it!)
4. Verify character PNGs display correctly (eye ball it?)
5. if needed or have time to.. optimize for performance

### Evaluation
1. idk yet

