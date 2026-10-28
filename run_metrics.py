import pandas as pd
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"c:\Users\megha\OneDrive\Desktop\Academics\Additional_project\Text-Emotion-Detection\Text Emotion Detection\data\emotion_dataset_raw.csv")
print("Total samples:", len(df))
print("\nClass distribution:")
print(df["Emotion"].value_counts())

df["Clean_Text"] = df["Text"].apply(nfx.remove_userhandles)
df["Clean_Text"] = df["Clean_Text"].apply(nfx.remove_stopwords)
df = df.dropna(subset=["Clean_Text"])
df = df[df["Clean_Text"].str.strip() != ""]

x = df["Clean_Text"]
y = df["Emotion"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nTrain: {len(x_train)}, Test: {len(x_test)}")
classes = sorted(y.unique())

# ----- Logistic Regression -----
pipe_lr = Pipeline([("cv", CountVectorizer()), ("lr", LogisticRegression(max_iter=1000))])
pipe_lr.fit(x_train, y_train)
y_pred_lr = pipe_lr.predict(x_test)
print("\n====== Logistic Regression ======")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr)*100:.2f}%")
print(f"Macro-F1: {f1_score(y_test, y_pred_lr, average='macro')*100:.2f}%")
print(f"Weighted-F1: {f1_score(y_test, y_pred_lr, average='weighted')*100:.2f}%")
print(classification_report(y_test, y_pred_lr, digits=4))

# ----- SVM (RBF, C=10) -----
pipe_svm = Pipeline([("cv", CountVectorizer()), ("svc", SVC(kernel="rbf", C=10))])
pipe_svm.fit(x_train, y_train)
y_pred_svm = pipe_svm.predict(x_test)
print("\n====== SVM (RBF, C=10) ======")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm)*100:.2f}%")
print(f"Macro-F1: {f1_score(y_test, y_pred_svm, average='macro')*100:.2f}%")
print(f"Weighted-F1: {f1_score(y_test, y_pred_svm, average='weighted')*100:.2f}%")
print(classification_report(y_test, y_pred_svm, digits=4))

# ----- Random Forest (100 estimators) -----
pipe_rf = Pipeline([("cv", CountVectorizer()), ("rf", RandomForestClassifier(n_estimators=100, random_state=42))])
pipe_rf.fit(x_train, y_train)
y_pred_rf = pipe_rf.predict(x_test)
print("\n====== Random Forest (100 estimators) ======")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)*100:.2f}%")
print(f"Macro-F1: {f1_score(y_test, y_pred_rf, average='macro')*100:.2f}%")
print(f"Weighted-F1: {f1_score(y_test, y_pred_rf, average='weighted')*100:.2f}%")
print(classification_report(y_test, y_pred_rf, digits=4))
