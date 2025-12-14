#ML training ian
#For explaining : Use Linreg instead of svm since faster (needed for live footage)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix as cm
from sklearn.linear_model import LogisticRegression as lgr
import os



def modtrain(csvpath, outputfile, testsize=0.2, randomstate=661) :

  df = pd.read_csv(csvpath)

  #USE EXACT FEATURE NAMES IN FEATURE EXTRACTION FROM FACE TRACKER
  dffeat = ["Smile", "L_Ear", "R_Ear", "L_Brow", "R_Brow", "Jaw", "Eardiff", "Pitch", "Yaw", "Mar"]

  # #safety net (AI Inclussion)
  # for c in dffeat + ["label"]:
  #   if c not in df.columns:
  #     print("ERROR missing column:", c)
  #     print("Found columns:", df.columns.tolist())
  #     return None

  # xs = df[dffeat].values.astype(float)
  xs = df[[c for c in df.columns if c.startswith("f_")]].values.astype(float)
  ys = df["label"].values

  #Splits data into train and test x is amount seen in feature, y is the type of emotion
  xtrain, xtest, ytrain, ytest = train_test_split(xs, ys, test_size=testsize, random_state=randomstate, stratify=ys)

  #log regression
  clf = lgr(max_iter=1500)
  clf.fit(xtrain,ytrain)

  ypred = clf.predict(xtest)



  #mod eval first classifcation report which shows how precise, second confus matrix basically TP FP TN FN
  print(cr(ytest, ypred))
  print(cm(ytest, ypred))

  #writes model into file to use in main can change to if difficulty with later, first is emtion, second which number determine which features, then intercepts
  if os.path.dirname(outputfile) != "":
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)

  with open(outputfile, "w") as fil:
    fil.write("Class " + " ".join(clf.classes_) + "\n")
    for r in clf.coef_:
      fil.write("Coef " + " ".join(map(str, r)) + "\n")
    fil.write("Intrcpt " + " ".join(map(str, clf.intercept_)) + "\n")

  print("Done")
  return clf

#load txt of log regression continues code abv and will be used in prediction of frame
def modloader(outmodtext):
  emo = []
  coe = []
  intc = []


  with open(outmodtext, "r") as fil:

    for l in fil:
      p = l.strip().split()
      if not p:
        continue
      txt = p.pop(0)

      if txt == "Class":
        emo = p

      elif txt == "Coef":
        cs = []
        for i in p:
          cs.append(float(i))
        coe.append(cs)

      elif txt == "Intrcpt":
        inters = []
        for i in p:
          inters.append(float(i))
        intc = inters

  emo = np.array(emo)
  coe = np.array(coe)
  intc = np.array(intc)

  return emo, coe, intc



#predicts frame uses from features list which has values that best match features. Uses numerics from modloader this will decide which PNG is used in a frame
def modpred(features, emo, coe, intc):
  sltcregskr = np.array([features]) @ coe.T + intc
  r = sltcregskr[0]
  bst = r[0]
  bsty = 0

  for i in range(1, len(r)):
    if r[i] > bst:
      bst = r[i]
      bsty = i

  return emo[bsty]


csv = "data/expressions.csv"          #adjust path to what is need i have my csv in folder called data
outmodtext = "model.txt"  #adjust path to what is need
# modtrain(csv, outmodtext)  # uncomment to train logistic regression model



#Neural network training (more advanced ML assited by AI)
#pickle like u guys want

from sklearn.neural_network import MLPClassifier as mlp
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def nntrain(csvpath, outputfile, testsize=0.2, randomstate=661) :

  df = pd.read_csv(csvpath)

  xs = df[[c for c in df.columns if c.startswith("f_")]].values.astype(float)
  ys = df["label"].values

  # Encode labels (string -> int)
  label_encoder = LabelEncoder()
  ys_encoded = label_encoder.fit_transform(ys)

  # Scale features for better neural net performance
  scaler = StandardScaler()
  xs_scaled = scaler.fit_transform(xs)

  xtrain, xtest, ytrain, ytest = train_test_split(xs_scaled, ys_encoded, test_size=testsize, random_state=randomstate, stratify=ys_encoded)

  #simple neural net (you can change hidden sizes later)
  net = mlp(hidden_layer_sizes=(64,64), max_iter=1000, random_state=randomstate)
  net.fit(xtrain,ytrain)

  ypred = net.predict(xtest)

  print(cr(ytest, ypred, target_names=label_encoder.classes_))
  print(cm(ytest, ypred))

  if os.path.dirname(outputfile) != "":
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)

  # Save as bundle with model, scaler, and label_encoder
  bundle = {
    "model": net,
    "scaler": scaler,
    "label_encoder": label_encoder
  }
  joblib.dump(bundle, outputfile)
  print("Done - saved bundle to", outputfile)
  return bundle


def nnload(modpath):
  return joblib.load(modpath)


def nnpred(features, bundle):
  scaler = bundle["scaler"]
  model = bundle["model"]
  label_encoder = bundle["label_encoder"]
  
  X_scaled = scaler.transform([features])
  y_pred = model.predict(X_scaled)[0]
  return label_encoder.inverse_transform([y_pred])[0]


if __name__ == "__main__":
  csv = "data/expressions.csv"
  fil = "expression_model.pkl"
  nntrain(csv, fil)
  # pred = nnpred(features, nnload(fil))