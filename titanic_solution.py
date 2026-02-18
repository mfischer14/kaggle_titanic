"""
Kaggle Titanic - Machine Learning from Disaster
Full ML pipeline: feature engineering + stacking ensemble → submission.csv

Usage:
    python titanic_solution.py

Expects train.csv and test.csv in the same directory.
Outputs submission.csv ready for Kaggle upload.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── 1. Load data ──────────────────────────────────────────────────────────────

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print(f"Train shape: {train.shape}  |  Test shape: {test.shape}")

# ── 2. Group survival rate (computed on raw data before feature engineering) ──
# For each passenger, what fraction of their ticket-mates survived?
# Train: leave-one-out to avoid label leakage within cross-validation.
# Test:  use all known train companions (no leakage risk).

ticket_stats = (
    train.groupby("Ticket")["Survived"]
    .agg(survived_sum="sum", total="count")
)


def loo_survival(row):
    """Leave-one-out group survival rate for a train passenger."""
    t = row["Ticket"]
    if t not in ticket_stats.index or ticket_stats.loc[t, "total"] < 2:
        return -1.0
    s = ticket_stats.loc[t, "survived_sum"] - row["Survived"]
    n = ticket_stats.loc[t, "total"] - 1
    return s / n


def test_group_survival(ticket):
    """Group survival rate for a test passenger using full train group."""
    if ticket not in ticket_stats.index:
        return -1.0
    return ticket_stats.loc[ticket, "survived_sum"] / ticket_stats.loc[ticket, "total"]


train["GroupSurvivalRate"] = train.apply(loo_survival, axis=1)
test["GroupSurvivalRate"]  = test["Ticket"].apply(test_group_survival)

# ── 2b. Family name survival rate ─────────────────────────────────────────────
# Passengers sharing a last name have correlated survival even across tickets.
# This extends coverage to family members who booked separately.

train["LastName"] = train["Name"].str.split(",").str[0].str.strip()
test["LastName"]  = test["Name"].str.split(",").str[0].str.strip()

name_stats = (
    train.groupby("LastName")["Survived"]
    .agg(survived_sum="sum", total="count")
)


def loo_name_survival(row):
    """Leave-one-out family-name survival rate for a train passenger."""
    n = row["LastName"]
    if n not in name_stats.index or name_stats.loc[n, "total"] < 2:
        return -1.0
    s = name_stats.loc[n, "survived_sum"] - row["Survived"]
    cnt = name_stats.loc[n, "total"] - 1
    return s / cnt


def test_name_survival(last_name):
    """Family-name survival rate for a test passenger."""
    if last_name not in name_stats.index:
        return -1.0
    return name_stats.loc[last_name, "survived_sum"] / name_stats.loc[last_name, "total"]


train["FamilyNameSurvivalRate"] = train.apply(loo_name_survival, axis=1)
test["FamilyNameSurvivalRate"]  = test["LastName"].apply(test_name_survival)

# Combined: take whichever signal is available (ticket > name > -1)
def combine_survival_rates(ticket_rate, name_rate):
    if ticket_rate != -1.0:
        return ticket_rate          # ticket is more specific
    return name_rate                # fall back to family name

train["BestGroupRate"] = [
    combine_survival_rates(t, n)
    for t, n in zip(train["GroupSurvivalRate"], train["FamilyNameSurvivalRate"])
]
test["BestGroupRate"] = [
    combine_survival_rates(t, n)
    for t, n in zip(test["GroupSurvivalRate"], test["FamilyNameSurvivalRate"])
]

# ── 3. Feature engineering ────────────────────────────────────────────────────

all_tickets       = pd.concat([train["Ticket"], test["Ticket"]])
ticket_group_size = all_tickets.value_counts().to_dict()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Title ──────────────────────────────────────────────────────────────
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title_map = {
        "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4,
        "Ms": 2, "Mme": 3, "Mlle": 2,
        "Dr": 5, "Rev": 5, "Col": 5, "Major": 5, "Capt": 5,
        "Lady": 5, "Sir": 5, "Don": 5, "Dona": 5,
        "Jonkheer": 5, "Countess": 5,
    }
    df["Title"] = df["Title"].map(title_map).fillna(5).astype(int)

    # ── Family ─────────────────────────────────────────────────────────────
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["FamilyGroup"] = pd.cut(
        df["FamilySize"], bins=[0, 1, 4, 20],
        labels=[0, 1, 2]   # 0=alone, 1=small, 2=large
    ).astype(int)
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # ── Ticket group ───────────────────────────────────────────────────────
    df["TicketGroupSize"] = df["Ticket"].map(ticket_group_size).fillna(1).astype(int)

    # ── Fare ───────────────────────────────────────────────────────────────
    df["Fare"] = df.groupby("Pclass")["Fare"].transform(
        lambda x: x.fillna(x.median())
    )
    df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"].clip(lower=1)
    df["FareBand"] = pd.qcut(df["Fare"], 4, labels=False, duplicates="drop")

    # ── Age ────────────────────────────────────────────────────────────────
    df["Age"] = df.groupby(["Title", "Pclass"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["IsChild"] = (df["Age"] < 15).astype(int)
    df["AgeBand"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100], labels=False)

    # ── Cabin / Deck ───────────────────────────────────────────────────────
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = (
        df["Cabin"].str[0]
        .map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8})
        .fillna(0)
        .astype(int)
    )

    # ── Embarked ───────────────────────────────────────────────────────────
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # ── Sex ────────────────────────────────────────────────────────────────
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # ── Interaction features ───────────────────────────────────────────────
    df["Sex_Pclass"]     = df["Sex"] * df["Pclass"]
    df["IsWomanOrChild"] = ((df["Sex"] == 1) | (df["IsChild"] == 1)).astype(int)

    return df


train = engineer_features(train)
test  = engineer_features(test)

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "FarePerPerson",
    "Embarked", "Title", "FamilySize", "FamilyGroup", "IsAlone",
    "TicketGroupSize", "FareBand", "AgeBand", "HasCabin", "Deck",
    "IsChild", "IsWomanOrChild", "Sex_Pclass",
    "GroupSurvivalRate",        # ticket-mate survival signal
    "FamilyNameSurvivalRate",   # name-based family survival signal
    "BestGroupRate",            # combined: ticket first, then name
]

X      = train[FEATURES]
y      = train["Survived"]
X_test = test[FEATURES]

print(f"Features used ({len(FEATURES)}): {FEATURES}")
print(f"X shape: {X.shape}  |  Missing values: {X.isnull().sum().sum()}")
ticket_cov = (train["GroupSurvivalRate"] != -1).mean()
name_cov   = (train["FamilyNameSurvivalRate"] != -1).mean()
best_cov   = (train["BestGroupRate"] != -1).mean()
print(f"Coverage — ticket: {ticket_cov:.1%}  name: {name_cov:.1%}  combined: {best_cov:.1%}")

# ── 4. Base models ────────────────────────────────────────────────────────────

rf = RandomForestClassifier(
    n_estimators=500, max_depth=6, min_samples_split=12,
    min_samples_leaf=4, max_features="sqrt", random_state=42, n_jobs=-1,
)

et = ExtraTreesClassifier(
    n_estimators=500, max_depth=7, min_samples_split=12,
    min_samples_leaf=4, max_features="sqrt", random_state=42, n_jobs=-1,
)

xgb = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    use_label_encoder=False, eval_metric="logloss",
    random_state=42, verbosity=0,
)

lgbm = LGBMClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

gb = GradientBoostingClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=6, random_state=42,
)

# ── 5. Stacking ensemble ──────────────────────────────────────────────────────

meta_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
])

stacking = StackingClassifier(
    estimators=[("rf", rf), ("et", et), ("xgb", xgb), ("lgbm", lgbm), ("gb", gb)],
    final_estimator=meta_lr,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    passthrough=False,
    n_jobs=-1,
)

# ── 6. Cross-validation ───────────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n── Cross-validation (5-fold accuracy) ──")
for name, model in [
    ("Random Forest", rf), ("Extra Trees", et),
    ("XGBoost", xgb), ("LightGBM", lgbm),
    ("Gradient Boosting", gb), ("Stacking", stacking),
]:
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  {name:<20} {scores.mean():.4f} ± {scores.std():.4f}")

# ── 7. Train final model & generate submission ────────────────────────────────

print("\nTraining stacking ensemble on full training set...")
stacking.fit(X, y)

preds = stacking.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived":    preds.astype(int),
})
submission.to_csv("submission.csv", index=False)

print(f"\nSubmission saved: submission.csv  ({len(submission)} rows)")
print(f"Predicted survivors: {preds.sum()} / {len(preds)}  "
      f"({preds.mean()*100:.1f}%)")
print("\nFirst 10 predictions:")
print(submission.head(10).to_string(index=False))
print("\nDone! Upload submission.csv to Kaggle.")
