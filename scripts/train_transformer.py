import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import joblib

from huggingface_hub import login
from dotenv import load_dotenv


# ============================================================
# 1. ENV TOKEN
# ============================================================

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception:
        print("⚠️ Token HF invalide ou absent.")
else:
    print("⚠️ Aucun token HuggingFace — push désactivé.")


# ============================================================
# 2. GPU AUTOMATIQUE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Device utilisé :", device)


# ============================================================
# 3. PATHS & CONFIG
# ============================================================

TRAIN_PATH = "data/processed/tickets_train.csv"
TEST_PATH = "data/processed/tickets_test.csv"
MODEL_NAME = "distilbert-base-multilingual-cased"
MODEL_DIR = "models/transformer"
HF_REPO_NAME = "callcenterai_mopls"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 4. FONCTION PRINCIPALE D’ENTRAÎNEMENT
# ============================================================

def run_training():
    # ----------- Chargement des données -----------
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("📌 Colonnes du dataset :", train_df.columns.tolist())

    if "Document" not in train_df.columns or "Topic_group" not in train_df.columns:
        raise ValueError("❌ Le CSV n'a pas les colonnes nécessaires.")

    # ----------- Équilibrage ----------- 
    min_samples = train_df["Topic_group"].value_counts().min()
    train_df = (
        train_df.groupby("Topic_group")
        .apply(lambda x: x.sample(min_samples, replace=True, random_state=42))
        .reset_index(drop=True)
    )

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    # ----------- Tokenizer -----------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        return tokenizer(batch["Document"], truncation=True, padding=False, max_length=128)

    train_ds = train_ds.map(preprocess, batched=True)
    test_ds = test_ds.map(preprocess, batched=True)

    # ----------- Labels -----------
    labels = sorted(train_df["Topic_group"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    def encode_label(example):
        return {"labels": label2id[example["Topic_group"]]}

    train_ds = train_ds.map(encode_label)
    test_ds = test_ds.map(encode_label)

    joblib.dump(label2id, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    # ----------- Modèle -----------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ----------- Metrics -----------
    def compute_metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    # ----------- MLflow -----------
    mlflow.set_experiment("transformer_multilang_v2")

    # ----------- Training arguments -----------
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=False,
        push_to_hub=False,
    )

    # ----------- Train ----------- 
    with mlflow.start_run():
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()

        results = trainer.evaluate()
        print("📊 Résultats :", results)
        mlflow.log_metrics(results)

        trainer.save_model(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

        mlflow.log_artifacts(MODEL_DIR, artifact_path="transformer_model")

    # ----------- HF Push -----------
    if HF_TOKEN:
        print("☁️ Upload HuggingFace…")
        model.push_to_hub(HF_REPO_NAME)
        tokenizer.push_to_hub(HF_REPO_NAME)
    else:
        print("⚠️ Aucun token — push ignoré.")

    print("🏁 Fin du training !")


# ============================================================
# 5. POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    run_training()
