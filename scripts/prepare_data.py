import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/all_tickets_processed_improved_v3.csv")

df = df.rename(columns={"Document": "Document", "Topic_group": "Topic_group"})

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Topic_group"])

train.to_csv("data/processed/tickets_train.csv", index=False)
test.to_csv("data/processed/tickets_test.csv", index=False)
