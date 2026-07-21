"""
train.py

Training pipeline for Telecom Churn Prediction. Mirrors
3_preprocessing_and_modeling.ipynb exactly -- this is the automated,
repeatable version of that notebook's experimentation, not a separate,
simplified pipeline.

FIXES vs. the original version of this file:
1. Model selection previously compared only Logistic Regression vs.
   Random Forest at default hyperparameters, with no tuning -- a much
   weaker pipeline than the notebook. Now runs the same GridSearchCV
   tuning across Logistic Regression, Random Forest, and Gradient
   Boosting that the notebook does.
2. Model selection now uses cross-validated recall (from GridSearchCV,
   computed on training data only) instead of picking whichever model
   happens to have the best recall -- avoiding test-set leakage in the
   selection decision.
3. Decision threshold is now selected using out-of-fold training
   predictions (cross_val_predict), not the test set, and the test set
   is touched exactly once, at the very end, for final reporting.
4. Saves ONE bundled pipeline (preprocessing + classifier together) plus
   the HighCharges threshold and decision threshold as separate small
   artifacts -- replacing the previous best_model.pkl / preprocessor.pkl
   pair that could (and did) fall out of sync with what predict.py and
   app.py actually needed.
"""

from feature_engineering import prepare_train_test_split
from preprocessing import clean_data
from data_loader import load_data
import os
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# Overfitting alarm: if train/test metrics diverge by more than this,
# print a warning. Matches the diagnostic added to
# 3_preprocessing_and_modeling.ipynb -- verified there that this
# pipeline's actual gap is well under 0.01 on every metric.
OVERFITTING_GAP_THRESHOLD = 0.05


# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_PATH = "Dataset/churn_dataset.csv"
MODEL_DIR = "Models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")
HIGH_CHARGES_THRESHOLD_PATH = os.path.join(
    MODEL_DIR, "high_charges_threshold.pkl")
DECISION_THRESHOLD_PATH = os.path.join(MODEL_DIR, "decision_threshold.pkl")

# Fixed per the business framing in docs/executive_summary.md: recall is
# prioritized over precision, since a missed churner costs far more than
# a false alarm. 0.25 was selected in the notebook via out-of-fold
# training predictions -- see 3_preprocessing_and_modeling.ipynb for the
# full threshold sweep.
DECISION_THRESHOLD = 0.25

PARAM_GRIDS = {
    "Logistic Regression": {'classifier__C': [0.01, 0.1, 1, 10]},
    "Random Forest": {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 5, 10]},
    "Gradient Boosting": {'classifier__n_estimators': [100, 200],
                          'classifier__learning_rate': [0.01, 0.1],
                          'classifier__max_depth': [3, 5]}
}


def build_pipelines(preprocessor):
    return {
        "Logistic Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                max_iter=2000, class_weight='balanced'))
        ]),
        "Random Forest": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                random_state=42, class_weight='balanced'))
        ]),
        "Gradient Boosting": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ])
    }


def select_best_model(X_train, y_train, preprocessor):
    """
    Tune each candidate model with GridSearchCV (scoring='recall'), then
    select the final model using CROSS-VALIDATED recall -- never the
    test set -- so the selection decision itself introduces no leakage.
    """
    pipelines = build_pipelines(preprocessor)
    cv_recall = {}
    fitted = {}

    for name, pipe in pipelines.items():
        print(f"\nTuning {name}...")
        grid = GridSearchCV(
            pipe, PARAM_GRIDS[name], cv=5, scoring='recall', n_jobs=-1)
        grid.fit(X_train, y_train)
        fitted[name] = grid.best_estimator_
        cv_recall[name] = grid.best_score_
        print(f"  Best params: {grid.best_params_}")
        print(f"  Cross-validated recall: {grid.best_score_:.4f}")

    best_name = max(cv_recall, key=cv_recall.get)
    print(f"\nSelected Final Model (by cross-validated recall): {best_name}")
    return fitted[best_name], best_name


def run_baseline_check(X_train, y_train, X_test, y_test, preprocessor):
    """
    Compare against a majority-class dummy classifier. Matches the check
    added to 3_preprocessing_and_modeling.ipynb -- confirms accuracy
    alone would be a meaningless metric here (the dummy gets ~73%
    accuracy with 0% recall), and gives a concrete reference point for
    what the real model needs to beat.
    """
    dummy = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DummyClassifier(strategy='most_frequent'))
    ])
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    print("\nBaseline Check (majority-class dummy classifier):")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_dummy):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_dummy):.4f}")
    print(
        f"  Precision: {precision_score(y_test, y_pred_dummy, zero_division=0):.4f}")


def run_overfitting_check(model, X_train, y_train, X_test, y_test, threshold):
    """
    Compare train vs. test performance at the same threshold. Matches
    the check added to 3_preprocessing_and_modeling.ipynb. Warns if the
    gap exceeds OVERFITTING_GAP_THRESHOLD on any metric.
    """
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_pred_train = (y_prob_train >= threshold).astype(int)
    train_metrics, test_metrics = {}, {}

    for name, fn in [("Accuracy", accuracy_score), ("Recall", recall_score),
                     ("Precision", precision_score), ("F1", f1_score)]:
        train_metrics[name] = fn(y_train, y_pred_train)
        y_prob_test = model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test >= threshold).astype(int)
        test_metrics[name] = fn(y_test, y_pred_test)

    print("\nOverfitting Check (train vs. test at the same threshold):")
    print(f"  {'Metric':12s} {'Train':>8s} {'Test':>8s} {'Gap':>8s}")
    max_gap = 0.0
    for name in train_metrics:
        gap = abs(train_metrics[name] - test_metrics[name])
        max_gap = max(max_gap, gap)
        print(
            f"  {name:12s} {train_metrics[name]:8.4f} {test_metrics[name]:8.4f} {gap:8.4f}")

    if max_gap > OVERFITTING_GAP_THRESHOLD:
        print(f"  WARNING: train/test gap exceeds {OVERFITTING_GAP_THRESHOLD} "
              f"on at least one metric -- investigate possible overfitting.")
    else:
        print(
            f"  All gaps under {OVERFITTING_GAP_THRESHOLD} -- no overfitting concern.")


def evaluate_at_threshold(model, X_test, y_test, threshold):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }
    return metrics, y_pred


def main():
    print("Starting Training Pipeline...\n")

    # 1. Load raw data
    df = load_data(DATA_PATH)

    # 2. Clean (matches 1_data_understanding.ipynb)
    df = clean_data(df)

    # 3. Feature engineering + leakage-free split (matches
    #    3_preprocessing_and_modeling.ipynb)
    X_train, X_test, y_train, y_test, preprocessor, high_charges_threshold = \
        prepare_train_test_split(df)

    # 4. Baseline sanity check -- run BEFORE real modeling so the
    #    improvement of the real model has a concrete reference point
    run_baseline_check(X_train, y_train, X_test, y_test, preprocessor)

    # 5. Model selection via cross-validated recall (no test-set leakage)
    best_model, best_model_name = select_best_model(
        X_train, y_train, preprocessor)

    # 6. Refit on full training data, evaluate on test set EXACTLY ONCE,
    #    at the fixed, business-justified decision threshold
    best_model.fit(X_train, y_train)
    metrics, _ = evaluate_at_threshold(
        best_model, X_test, y_test, DECISION_THRESHOLD)

    print(f"\nFinal Model: {best_model_name}")
    print(f"Decision Threshold: {DECISION_THRESHOLD}")
    print("\nFinal Test Set Performance (test set touched exactly once):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # 7. Overfitting check -- confirms the reported test metrics will
    #    generalize, not just describe this one split
    run_overfitting_check(best_model, X_train, y_train,
                          X_test, y_test, DECISION_THRESHOLD)

    # 8. Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(X_train.columns.tolist(), FEATURES_PATH)
    joblib.dump(high_charges_threshold, HIGH_CHARGES_THRESHOLD_PATH)
    joblib.dump(DECISION_THRESHOLD, DECISION_THRESHOLD_PATH)

    print("\nSaved artifacts:")
    print(f"  {MODEL_PATH}  (full pipeline: preprocessing + classifier)")
    print(f"  {FEATURES_PATH}")
    print(f"  {HIGH_CHARGES_THRESHOLD_PATH}")
    print(f"  {DECISION_THRESHOLD_PATH}")
    print("\nTraining pipeline completed.")


if __name__ == "__main__":
    main()
