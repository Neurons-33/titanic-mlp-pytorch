import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from src.model import Titanic
from src.features import fit_preprocessor, transform_with_artifacts
from src.train import train_model


# -------------------
# Global config
# -------------------
SEED = 42
HIDDEN = 32
EPOCHS = 600
TEST_SIZE = 0.2
LR = 3e-4
DEPTH = 2
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

ENSEMBLE_SEEDS = [11, 22, 33, 44, 55]
THRESHOLD = 0.5


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------
# Paths
# -------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
TRAIN_PATH = os.path.join(ROOT, "data", "raw", "train.csv")
TEST_PATH = os.path.join(ROOT, "data", "raw", "test.csv")

OUT_DIR = os.path.join(ROOT, "outputs", "submissions")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")


# -------------------
# Load data
# -------------------
df = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

# -------------------
# Fit preprocess on TRAIN only
# -------------------
X_all, artifacts = fit_preprocessor(df)
y_all = df["Survived"].values.reshape(-1, 1).astype(np.float32)

# split for optional val (training loop prints val loss)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=SEED
)

X_train_t = torch.tensor(X_tr, dtype=torch.float32)
y_train_t = torch.tensor(y_tr, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

# -------------------
# Prepare TEST features using artifacts
# -------------------
X_test_final = transform_with_artifacts(df_test, artifacts)
X_test_t = torch.tensor(X_test_final, dtype=torch.float32)
passenger_id = df_test["PassengerId"].values

# -------------------
# Train + Ensemble inference
# -------------------
set_seed(SEED)
print(f"HIDDEN={HIDDEN} DEPTH={DEPTH} LR={LR} DROPOUT={DROPOUT} EPOCHS={EPOCHS}")

probs_list = []
for s in ENSEMBLE_SEEDS:
    model, _ = train_model(
        X_train_t, y_train_t,
        X_val_t, y_val_t,
        epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        seed=s,
        depth=DEPTH,
        lr=LR,
        hidden=HIDDEN,
        dropout=DROPOUT,
        log_every=100
    )
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t)).squeeze().cpu().numpy()
    probs_list.append(probs)

probs_ensemble = np.mean(probs_list, axis=0)
y_test_pred = (probs_ensemble >= THRESHOLD).astype(int)

# -------------------
# Save submission
# -------------------
submission = pd.DataFrame({
    "PassengerId": passenger_id,
    "Survived": y_test_pred
})
submission.to_csv(OUT_PATH, index=False)
print("saved:", OUT_PATH)

print("X_train:", X_all.shape)
print("X_test :", X_test_final.shape)