import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ======== Enhanced CORAL Implementation ========
class CORAL_Ordinal:
    """Balanced + Vectorized CORAL for tabular features."""
    def __init__(self, n_classes=5, C=1.0, max_iter=3000, random_state=42):
        self.n_classes = n_classes
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.models = []
        self.scaler = None

    def fit(self, X, y):
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        self.models = []
        for k in range(1, self.n_classes):
            y_bin = (y >= k).astype(int)
            clf = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver="lbfgs",
                random_state=self.random_state,
                class_weight="balanced"  # ✅ balance rare stages
            )
            clf.fit(Xs, y_bin)
            self.models.append(clf)
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        cum_probs = np.column_stack([clf.predict_proba(Xs)[:, 1] for clf in self.models])
        probs = np.zeros((X.shape[0], self.n_classes))
        probs[:, 0] = 1 - cum_probs[:, 0]
        for k in range(1, self.n_classes - 1):
            probs[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
        probs[:, -1] = cum_probs[:, -1]
        return probs

    def predict(self, X, threshold=0.5):
        Xs = self.scaler.transform(X)
        cum_probs = np.column_stack([clf.predict_proba(Xs)[:, 1] for clf in self.models])
        preds = np.sum(cum_probs >= threshold, axis=1)
        return preds


# ======== Helper: Find Best Threshold (Optional) ========
def find_best_threshold(model, X_val, y_val):
    thresholds = np.linspace(0.3, 0.7, 9)
    best_t, best_kappa = 0.5, -1
    for t in thresholds:
        preds = model.predict(X_val, threshold=t)
        kappa = cohen_kappa_score(y_val, preds, weights="quadratic")
        if kappa > best_kappa:
            best_kappa, best_t = kappa, t
    return best_t


# ======== CV Validation (Balanced + RepeatedStratifiedKFold) ========
def coral_cv(df, n_splits=3, n_repeats=5, C=1.0, search_t=False):
    X = df.drop(columns=["ID", "Stage", "Stage_num"]).fillna(0)
    y = df["Stage_num"].astype(int)
    n_classes = len(np.unique(y))

    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    accs, kappas = [], []
    cm_sum = np.zeros((n_classes, n_classes))

    print(f"🚀 Running {n_splits}-Fold × {n_repeats}-Repeat CV (Balanced CORAL)...")
    for fold, (tr_idx, va_idx) in enumerate(rkf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = CORAL_Ordinal(n_classes=n_classes, C=C)
        model.fit(X_tr.values, y_tr.values)

        # 可选：调节 threshold 提升 QWK
        threshold = find_best_threshold(model, X_va.values, y_va.values) if search_t else 0.5
        pred = model.predict(X_va.values, threshold=threshold)

        acc = accuracy_score(y_va, pred)
        qwk = cohen_kappa_score(y_va, pred, weights="quadratic")
        accs.append(acc)
        kappas.append(qwk)

        cm = confusion_matrix(y_va, pred, labels=range(n_classes), normalize="true")
        cm_sum += cm

        print(f"Fold {fold}: Acc={acc:.3f}, QWK={qwk:.3f}, Thr={threshold:.2f}")

    cm_avg = cm_sum / (n_splits * n_repeats)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_avg, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=[f"Pred F{i}" for i in range(n_classes)],
                yticklabels=[f"True F{i}" for i in range(n_classes)])
    plt.title(f"Average Normalized Confusion Matrix ({n_splits}×{n_repeats} CV)")
    plt.xlabel("Predicted Stage")
    plt.ylabel("True Stage")
    plt.tight_layout()
    plt.show()

    print(f"\n📊 CV Results (Balanced CORAL):")
    print(f"Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Quadratic Weighted Kappa: {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")
    return np.mean(accs), np.mean(kappas)


# ======== Train-Test Evaluation ========
def coral_train_test(df_train, df_test, C=1.0, show_cm=True, search_t=False):
    X_tr = df_train.drop(columns=["ID", "Stage", "Stage_num"]).fillna(0)
    y_tr = df_train["Stage_num"].astype(int)
    X_te = df_test.drop(columns=["ID", "Stage", "Stage_num"]).fillna(0)
    y_te = df_test["Stage_num"].astype(int)

    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    n_classes = len(np.unique(y_tr))

    model = CORAL_Ordinal(n_classes=n_classes, C=C)
    model.fit(X_tr.values, y_tr.values)

    threshold = find_best_threshold(model, X_te.values, y_te.values) if search_t else 0.5
    y_pred = model.predict(X_te.values, threshold=threshold)

    acc = accuracy_score(y_te, y_pred)
    qwk = cohen_kappa_score(y_te, y_pred, weights="quadratic")

    print(f"\n📊 Test Set Results (Balanced CORAL):")
    print(f"Accuracy = {acc:.10f}")
    print(f"Quadratic Weighted Kappa = {qwk:.3f} (threshold={threshold:.2f})")

    if show_cm:
        labels = sorted(np.unique(np.concatenate([y_tr, y_te])))
        cm = confusion_matrix(y_te, y_pred, labels=labels, normalize="true")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=[f"Pred F{int(i)}" for i in labels],
                    yticklabels=[f"True F{int(i)}" for i in labels])
        plt.title("Normalized Confusion Matrix – Balanced CORAL (Test Set)")
        plt.xlabel("Predicted Stage")
        plt.ylabel("True Stage")
        plt.tight_layout()
        plt.show()

    return model, acc, qwk
