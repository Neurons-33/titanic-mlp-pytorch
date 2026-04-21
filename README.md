# Titanic - PyTorch MLP (Kaggle)

A reproducible PyTorch MLP pipeline for the Kaggle Titanic dataset.

This project demonstrates:
- Modular feature engineering
- Clean training loop separation
- Ensemble inference
- Reproducible submission generation

[![View Report](https://img.shields.io/badge/Technical_Report-View_PDF-blue?style=for-the-badge&logo=adobeacrobatreader)](https://neurons-33.github.io/titanic-mlp-pytorch/reports/docs/titanic_validation_loop_v1.pdf)
---

## Project Structure

- src/
    - model.py # MLP architecture definition
    - train.py # training loop (optimizer, loss, logging)
    - features.py # feature engineering (fit / transform pipeline)
- scripts/
    - make_submission.py # production entry point
- data/
    - raw/
        - train.csv
        - test.csv
- outputs/
    - submission/
        - submission.csv
- requirements.txt
- README.md


---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt

```

---

## Data

- Download Titanic dataset from Kaggle:
[titanic](https://www.kaggle.com/competitions/titanic)

- Place the files here:
data/raw/train.csv
data/raw/test.csv

---

## Run Submission Pipeline

- From project root:
python scripts/make_submission.py

- Output file:
outputs/submissions/submission.csv

---

## Feature Pipeline

- Preprocessing is fitted on training data only.
- All statistics (median, quantile bins, encoders) are stored as artifacts.
- The same artifacts are reused for validation and test transformation.

- This ensures:
No data leakage
Stable feature space
Reproducible inference

---

## Model

- PyTorch MLP
- Configurable depth
- Dropout regularization
- Adam optimizer
- Ensemble across multiple random seeds

---

## Reproducibility

- Randomness is controlled via:
Numpy seed
PyTorch seed
Deterministic train / validation split

---

## Notes

- The submission script is intentionally clean (no plotting or experiment logic).
