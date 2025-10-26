```python
#for creating, training and saving the model- shape detection

import os, glob, argparse, joblib, numpy as np, pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

RANDOM_STATE = 42

BASE = ["area","perimeter","circularity","aspect_ratio","solidity"]
ROBUST = ["vertices_hull","extent","eq_diameter","eccentricity",
          "hu1","hu2","hu3","hu4","hu5","hu6","hu7"]

def choose_features(df: pd.DataFrame):
    feats = [c for c in BASE if c in df.columns] + [c for c in ROBUST if c in df.columns]
    missing_base = [c for c in BASE if c not in feats]
    if missing_base:
        raise ValueError(f"CSV missing required base features: {missing_base}")
    return feats

def load_many(csvdir: str, pattern: str = "*.csv") -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(csvdir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {csvdir!r}")
    dfs=[]
    for p in paths:
        df = pd.read_csv(p)
        need = ["image_id","label"]
        for c in need:
            if c not in df.columns:
                raise ValueError(f"{p} missing column: {c}")
        dfs.append(df.copy())
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return all_df

def load_single(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain image_id and label")
    return df

def build_models():
    pipelines = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "svc_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", probability=False,
                        cache_size=500, random_state=RANDOM_STATE))
        ]),
        "rf": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"))
        ]),
        "gb": Pipeline([
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
        ]),
    }
    grids = {
        "logreg": {"clf__C":[0.1,1.0,3.0,10.0], "clf__penalty":["l2"]},
        "svc_rbf": {"clf__C":[0.5,1.0,3.0,10.0], "clf__gamma":["scale",0.1,0.01]},
        "rf": {"clf__n_estimators":[200,300,500], "clf__max_depth":[None,8,12], "clf__min_samples_split":[2,5]},
        "gb": {"clf__n_estimators":[150,300], "clf__learning_rate":[0.05,0.1,0.2], "clf__max_depth":[2,3]},
    }
    return pipelines, grids

def main(args):
    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.csvdir:
        df = load_many(args.csvdir, args.pattern)
    else:
        df = load_single(args.csv)

    feature_cols = choose_features(df)

    X = df[feature_cols]
    y_str = df["label"].astype(str).values
    groups_all = df["image_id"].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes = list(le.classes_)

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups_all, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedGroupKFold(n_splits=args.cv_splits, shuffle=True, random_state=RANDOM_STATE)
    pipes, grids = build_models()

    best_name, best_score, best_est, best_params = None, -np.inf, None, None
    for name, pipe in pipes.items():
        gs = GridSearchCV(
            pipe, grids[name], scoring="accuracy",
            cv=cv.split(X_train, y_train, groups=groups_train),
            n_jobs=4, refit=True, verbose=0, error_score='raise', pre_dispatch='1*n_jobs'
        )
        gs.fit(X_train, y_train)
        if gs.best_score_ > best_score:
            best_name, best_score, best_est, best_params = name, gs.best_score_, gs.best_estimator_, gs.best_params_

    print(f"\n>>> Selected model: {best_name} with CV acc {best_score:.4f}")
    print("    Params:", best_params)

    if best_name == "svc_rbf" and args.refit_prob_svc:
        from sklearn.base import clone
        svc2 = clone(best_est); svc2.set_params(clf__probability=True); svc2.fit(X_train, y_train)
        best_est = svc2

    y_pred = best_est.predict(X_test)
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=pred):"); print(cm)

    joblib.dump(best_est, Path(out_dir, "best_shape_clf.joblib"))
    joblib.dump(le, Path(out_dir, "label_encoder.joblib"))
    print(f"\nSaved model → {Path(out_dir,'best_shape_clf.joblib')}")
    print(f"Saved label encoder → {Path(out_dir,'label_encoder.joblib')}")
    print("Feature columns used:", feature_cols)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="ml_shapes_robust/shapes_features.csv") 
    parser.add_argument("--csvdir", default="csvdata")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--outdir", default="models")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--refit_prob_svc", action="store_true")
    args = parser.parse_args()
    if not args.csvdir or args.csvdir.strip()=="":
        args.csvdir = None
    main(args)
