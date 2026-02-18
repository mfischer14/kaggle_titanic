# Kaggle Titanic – ML from Disaster

Binary classification: predict which passengers survived the Titanic.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download the data from the [Kaggle competition page](https://www.kaggle.com/competitions/titanic/data) and place the files in this directory:

```
kaggle_titanic/
├── train.csv
├── test.csv
└── gender_submission.csv   # optional sample submission
```

Or use the Kaggle CLI:

```bash
pip install kaggle
kaggle competitions download -c titanic
unzip titanic.zip
```

## Run

```bash
python titanic_solution.py
```

This will:
1. Load and engineer features from `train.csv` / `test.csv`
2. Train a soft-voting ensemble (Random Forest + XGBoost + Gradient Boosting + Logistic Regression)
3. Print 5-fold cross-validation accuracy for each model
4. Save `submission.csv` (ready for Kaggle upload)
5. Save `feature_importances.png`

## Pipeline overview

| Step | Details |
|---|---|
| Title extraction | Parsed from `Name`; rare titles grouped |
| Age imputation | Median by Title × Pclass group |
| Fare imputation | Median by Pclass |
| Family size | `SibSp + Parch + 1`; `IsAlone` flag |
| Cabin | Binary known/unknown flag |
| Binning | `AgeBand` (5 bins), `FareBand` (quartiles) |
| Ensemble | RF × 3 + XGB × 3 + GB × 2 + LR × 1 (soft vote) |

## Expected CV accuracy

~82–84% (5-fold stratified) with the default ensemble.

## Submit to Kaggle

1. Go to [Submit Predictions](https://www.kaggle.com/competitions/titanic/submit)
2. Upload `submission.csv`
3. Check your score on the leaderboard
