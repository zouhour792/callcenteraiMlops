import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# === Charger les données ===
train = pd.read_csv("data/processed/tickets_train.csv")
test = pd.read_csv("data/processed/tickets_test.csv")

# === Correction des colonnes ===
X_train, y_train = train["Document"], train["Topic_group"]
X_test, y_test = test["Document"], test["Topic_group"]

# === Label Encoder ===
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# === MLflow Tracking ===
mlflow.set_experiment("tfidf_svm_experiment")

with mlflow.start_run():

    # === TF-IDF Vectorizer ===
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50000,
        ngram_range=(1,2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # === Modèle SVM calibré ===
    svm = LinearSVC()
    clf = CalibratedClassifierCV(svm)

    clf.fit(X_train_vec, y_train_enc)

    # === Prédictions ===
    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, average="macro")

    print("Accuracy :", acc)
    print("F1 Macro :", f1)

    # === Logging MLflow ===
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1)

    # === Sauvegarde modèles ===
    os.makedirs("models/tfidf", exist_ok=True)

    joblib.dump(vectorizer, "models/tfidf/vectorizer.joblib")
    joblib.dump(clf, "models/tfidf/model.joblib")
    joblib.dump(label_encoder, "models/tfidf/label_encoder.joblib")

    # MLflow Registry
    mlflow.sklearn.log_model(clf, "tfidf_model")

print("🎉 Modèle TF-IDF + SVM entraîné et sauvegardé   !")
print("👉 Fichiers générés dans models/tfidf/")
