### Collect Training Data (yo face)
record one emotion at a time.

**Run:**

```
python build_dataset.py
```

When it asks for a label, type one of:

- happy
- sad
- excited
- angry
- shocked

A webcam window will open.

- Press **R** to start recording.
- Hold the expression still and clear for a few seconds.
- Press **R** again to stop recording.
- Repeat until you collect about 150 to 200 frames for that emotion.
- Press **Q** to quit.

Repeat this entire process for all five emotions or more later on when we make this better.

All data is saved automatically into:

```
data/expressions.csv
```

it saves each recording in same file

### Train the Model (In Google Colab)

Upload:

```
data/expressions.csv
```

to Colab.

Run the training code, which:

- Loads the CSV
- Trains logistic regression
- Prints accuracy and confusion matrix
- Saves: `expression_model.pkl`

Download `expression_model.pkl`.

Put it into your project folder:

```
pngvtuber/expression_model.pkl
```

### Monkey Images

Inside the `monkeyfaces/` folder, add more if we need to

### Run the Live PNG-VTuber

```
python main.py
```

- Left: your webcam video
- Top left: live predicted emotion
- Right: a monkey image that changes with your emotion

Press **Q** to quit.

### each file stuff

- **face_tracker.py**  
  Detects your face and outputs 468 landmarks.

- **feature_extractor.py**  
  Converts landmarks into numeric features for machine learning.

- **build_dataset.py**  
  Records expressions and saves them to CSV.

- **Training in Colab**  
  Trains the logistic regression model.

- **action_classifier.py**  
  Loads the trained model and predicts emotions live.

- **main.py**  
  Runs everything in real time and shows the monkey avatar.

### improve our model cuz right now its not very confident/accurate when moving 
...