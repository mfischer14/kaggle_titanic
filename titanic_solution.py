"""
Kaggle Titanic - Machine Learning from Disaster
Full ML pipeline: feature engineering + ensemble model → submission.csv

Usage:
    python titanic_solution.py

Expects train.csv and test.csv in the same directory.
Outputs submission.csv ready for Kaggle upload.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ── 1. Load data ──────────────────────────────────────────────────────────────

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print(f"Train shape: {train.shape}  |  Test shape: {test.shape}")
print("\nTrain sample:")
print(train.head(3).to_string())

# ── 2. Feature engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Title extracted from Name
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    title_map = {
        "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5,
        "Ms": 2, "Mme": 3, "Mlle": 2, "Dr": 5, "Rev": 5,
        "Col": 5, "Major": 5, "Capt": 5, "Lady": 5, "Sir": 5,
        "Don": 5, "Dona": 5, "Jonkheer": 5, "Countess": 5,
    }
    df["Title"] = df["Title"].map(title_map).fillna(5).astype(int)

    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Fare: fill missing (test set has 1) with median per class
    df["Fare"] = df.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.median()))
    df["FareBand"] = pd.qcut(df["Fare"], 4, labels=False, duplicates="drop")

    # Age: impute with median per Title × Pclass
    df["Age"] = df.groupby(["Title", "Pclass"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Age"] = df["Age"].fillna(df["Age"].median())  # fallback
    df["AgeBand"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100], labels=False)

    # Cabin: known vs unknown
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # Embarked: fill 2 missing with mode
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    embarked_map = {"S": 0, "C": 1, "Q": 2}
    df["Embarked"] = df["Embarked"].map(embarked_map)

    # Sex to numeric
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    return df


train = engineer_features(train)
test  = engineer_features(test)

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked", "Title", "FamilySize", "IsAlone",
    "FareBand", "AgeBand", "HasCabin",
]

X      = train[FEATURES]
y      = train["Survived"]
X_test = test[FEATURES]

print(f"\nFeatures used: {FEATURES}")
print(f"X shape: {X.shape}  |  Missing values: {X.isnull().sum().sum()}")

# ── 3. Models ─────────────────────────────────────────────────────────────────

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

gb = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
])

# Soft-voting ensemble (averages predicted probabilities)
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("gb", gb), ("lr", lr)],
    voting="soft",
    weights=[3, 3, 2, 1],
    n_jobs=-1,
)

# ── 4. Cross-validation ───────────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n── Cross-validation (5-fold accuracy) ──")
for name, model in [("Random Forest", rf), ("XGBoost", xgb),
                     ("Gradient Boosting", gb), ("Logistic Reg", lr),
                     ("Ensemble", ensemble)]:
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  {name:<20} {scores.mean():.4f} ± {scores.std():.4f}")

# ── 5. Train final model on all training data ─────────────────────────────────

print("\nTraining final ensemble on full training set...")
ensemble.fit(X, y)

# ── 6. Feature importance (from Random Forest component) ─────────────────────

rf.fit(X, y)  # already fitted inside ensemble but fit standalone for importances
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

print("\n── Feature importances (Random Forest) ──")
print(importances.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Feature Importances (Random Forest)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
print("Saved: feature_importances.png")

# ── 7. Generate predictions & submission file ─────────────────────────────────

preds = ensemble.predict(X_test)

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
