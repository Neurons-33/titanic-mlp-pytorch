import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def fit_preprocessor(train_df: pd.DataFrame):
    """
    Fit feature pipeline on TRAIN only.
    Returns:
      - X_train_final (np.ndarray)
      - artifacts (dict): objects & stats needed to transform test
    """
    X_train = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Ticket']].copy()

    # --- fill missing (TRAIN stats)
    age_median = X_train['Age'].median()
    X_train['Age'] = X_train['Age'].fillna(age_median)

    fare_median = X_train['Fare'].median()
    X_train['Fare'] = X_train['Fare'].fillna(fare_median)

    # --- feature engineering
    X_train['Sex_Pclass'] = X_train['Sex'] + '_P' + X_train['Pclass'].astype(str)

    ticket_counts = X_train['Ticket'].value_counts()
    X_train['TicketGroupSize'] = X_train['Ticket'].map(ticket_counts).fillna(1).astype(int)

    bins = [0, 2, 4, float('inf')]
    grp_labels = ['solo', 'small', 'medium']
    X_train['IsGroupTicket'] = pd.cut(
        X_train['TicketGroupSize'], bins=bins, labels=grp_labels, include_lowest=True
    )

    q = np.quantile(X_train['Fare'], [0, 0.25, 0.5, 0.75, 1.0])
    q = np.unique(q)
    if len(q) < 2:
        raise ValueError("Fare quantile bins collapsed (q has < 2 unique edges).")

    fare_labels = [f"Q{i}" for i in range(len(q) - 1)]
    X_train['FareBin'] = pd.cut(X_train['Fare'], bins=q, labels=fare_labels, include_lowest=True)

    X_train['GroupTicket_Fare'] = (
        'G' + X_train['IsGroupTicket'].astype(str) + '_' + X_train['FareBin'].astype(str)
    )

    # --- select cols
    num_columns = ['Age']
    cat_columns = ['Sex_Pclass', 'GroupTicket_Fare']

    # --- fit transformers (TRAIN only)
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_columns])

    one = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = one.fit_transform(X_train[cat_columns])

    X_train_final = np.hstack([X_train_num, X_train_cat])

    artifacts = {
        "age_median": float(age_median),
        "fare_median": float(fare_median),
        "ticket_counts": ticket_counts,
        "bins": bins,
        "grp_labels": grp_labels,
        "q": q,
        "fare_labels": fare_labels,
        "num_columns": num_columns,
        "cat_columns": cat_columns,
        "scaler": scaler,
        "onehot": one,
    }
    return X_train_final, artifacts


def transform_with_artifacts(df: pd.DataFrame, artifacts: dict):
    """
    Transform any df (val/test) using TRAIN-fitted artifacts.
    Returns X_final (np.ndarray).
    """
    X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Ticket']].copy()

    # fill missing using TRAIN stats
    X['Age'] = X['Age'].fillna(artifacts["age_median"])
    X['Fare'] = X['Fare'].fillna(artifacts["fare_median"])

    # feature engineering (mirror)
    X['Sex_Pclass'] = X['Sex'] + '_P' + X['Pclass'].astype(str)

    ticket_counts = artifacts["ticket_counts"]
    X['TicketGroupSize'] = X['Ticket'].map(ticket_counts).fillna(1).astype(int)

    X['IsGroupTicket'] = pd.cut(
        X['TicketGroupSize'],
        bins=artifacts["bins"],
        labels=artifacts["grp_labels"],
        include_lowest=True
    )

    X['FareBin'] = pd.cut(
        X['Fare'],
        bins=artifacts["q"],
        labels=artifacts["fare_labels"],
        include_lowest=True
    )

    X['GroupTicket_Fare'] = (
        'G' + X['IsGroupTicket'].astype(str) + '_' + X['FareBin'].astype(str)
    )

    # numeric / categorical
    scaler = artifacts["scaler"]
    one = artifacts["onehot"]
    num_columns = artifacts["num_columns"]
    cat_columns = artifacts["cat_columns"]

    X_num = scaler.transform(X[num_columns])
    X_cat = one.transform(X[cat_columns])

    X_final = np.hstack([X_num, X_cat])
    return X_final