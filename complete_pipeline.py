"""
Complete Credit Risk Prediction Pipeline
========================================
Covers:
  1. Data loading & preprocessing
  2. Random Forest training with class balancing
  3. Expected Loss threshold optimization
  4. Fairness analysis & fix (reweighting + per-group thresholds)
  5. Full visualization & reporting

Run:
    pip install scikit-learn matplotlib seaborn pandas numpy
    python complete_pipeline.py

Optional: place cs-training.csv in ./data/ for real dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)


# ═══════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════

def load_data(path="data/cs-training.csv"):
    """Load real dataset or generate synthetic data."""
    try:
        df = pd.read_csv(path, index_col=0)
        print(f"✓ Loaded real dataset: {df.shape}")
    except FileNotFoundError:
        print("ℹ  Real dataset not found → generating synthetic data.")
        df = generate_synthetic_data()
    return df


def generate_synthetic_data(n=10000, random_state=42):
    """
    Synthetic data mimicking Give Me Some Credit schema.
    Includes simulated loan amounts (not in original dataset).
    """
    rng = np.random.default_rng(random_state)

    age             = rng.integers(18, 80, size=n)
    revolving_util  = rng.beta(2, 5, size=n).clip(0, 1)
    debt_ratio      = rng.beta(1, 4, size=n).clip(0, 1)
    monthly_income  = rng.lognormal(mean=8.5, sigma=0.6, size=n)
    num_open_credit = rng.integers(0, 20, size=n)
    num_real_estate = rng.integers(0, 4, size=n)
    num_dependents  = rng.integers(0, 6, size=n).astype(float)
    late_30_59      = rng.integers(0, 5, size=n)
    late_60_89      = rng.integers(0, 3, size=n)
    late_90         = rng.integers(0, 3, size=n)

    # Default probability influenced by features
    default_logit = (
        -3.5
        + 2.0 * revolving_util
        + 1.5 * (late_90 > 0).astype(float)
        + 1.0 * (late_30_59 > 2).astype(float)
        + 0.5 * debt_ratio
        - 0.02 * (age - 40)
        - 0.00005 * monthly_income
    )
    default_prob = 1 / (1 + np.exp(-default_logit))
    default      = rng.binomial(1, default_prob)

    # Simulate loan amount: higher income + more credit lines = bigger loan
    monthly_income_raw  = monthly_income.copy()
    base_loan           = monthly_income_raw * rng.uniform(3, 18, size=n)
    credit_factor       = 1 + 0.05 * num_open_credit
    real_estate_factor  = 1 + 0.30 * num_real_estate
    loan_amount         = (base_loan * credit_factor * real_estate_factor).clip(5000, 2000000)
    loan_amount         = np.round(loan_amount, -2)

    # Introduce missing values (realistic)
    monthly_income = monthly_income.astype(float)
    monthly_income[rng.choice(n, size=int(0.10 * n), replace=False)] = np.nan
    num_dependents[rng.choice(n, size=int(0.05 * n), replace=False)] = np.nan

    df = pd.DataFrame({
        "SeriousDlqin2yrs"                          : default,
        "RevolvingUtilizationOfUnsecuredLines"       : revolving_util,
        "age"                                        : age,
        "NumberOfTime30-59DaysPastDueNotWorse"       : late_30_59,
        "DebtRatio"                                  : debt_ratio,
        "MonthlyIncome"                              : monthly_income,
        "NumberOfOpenCreditLinesAndLoans"            : num_open_credit,
        "NumberOfTimes90DaysLate"                    : late_90,
        "NumberRealEstateLoansOrLines"               : num_real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse"       : late_60_89,
        "NumberOfDependents"                         : num_dependents,
        "LoanAmount"                                 : loan_amount,
    })

    print(f"✓ Synthetic data generated: {df.shape}")
    print(f"  Default rate : {df['SeriousDlqin2yrs'].mean():.2%}")
    print(f"  Loan range   : ₹{loan_amount.min():,.0f} → ₹{loan_amount.max():,.0f}")
    return df


# ═══════════════════════════════════════════════════════
# STEP 2 — CLEAN & PREPROCESS
# ═══════════════════════════════════════════════════════

def preprocess(df):
    """
    Clean data:
      - Cap outliers at 99th percentile
      - Engineer TotalLatePayments feature
      - Create AgeGroup bins for fairness analysis
    """
    df = df.copy()

    feature_cols = [
        "RevolvingUtilizationOfUnsecuredLines", "age",
        "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
        "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
    ]

    # Cap outliers
    for col in feature_cols:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    # Feature engineering
    df["TotalLatePayments"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    feature_cols.append("TotalLatePayments")

    # Age groups for fairness
    df["AgeGroup"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 100],
        labels=["Young (<30)", "Middle (30-45)", "Senior (45-60)", "Elder (60+)"]
    )

    X          = df[feature_cols]
    y          = df["SeriousDlqin2yrs"]
    age_group  = df["AgeGroup"]
    loan_amt   = df["LoanAmount"]

    print(f"\n── Preprocessing Summary ──")
    print(f"  Features  : {len(feature_cols)}")
    print(f"  Samples   : {len(df):,}")
    print(f"  Default % : {y.mean():.2%}")

    return X, y, age_group, loan_amt, feature_cols


# ═══════════════════════════════════════════════════════
# STEP 3 — TRAIN RANDOM FOREST
# ═══════════════════════════════════════════════════════

def train_model(X_train, y_train, ag_train, use_reweighting=True):
    """
    Train Random Forest inside a sklearn Pipeline.

    Pipeline = Imputer → Scaler → RandomForest
    Why Pipeline? Imputer and Scaler only learn from training data.
    Prevents data leakage into test set.

    class_weight handled via sample_weight when use_reweighting=True
    (combines class imbalance fix + fairness reweighting in one step)
    """
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),  # fill missing with median
        ("sc",  StandardScaler()),                   # normalise all features
        ("clf", RandomForestClassifier(
            n_estimators=200,      # 200 trees vote
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ))
    ])

    if use_reweighting:
        sample_weights = compute_sample_weights(y_train, ag_train)
        pipe.fit(X_train, y_train, clf__sample_weight=sample_weights)
        print("✓ Model trained with sample reweighting (class + group fairness)")
    else:
        pipe.fit(X_train, y_train)
        print("✓ Model trained (no reweighting)")

    return pipe


def compute_sample_weights(y_train, age_group_train):
    """
    Combine two weight signals:
      1. Class weight    → defaulters weighted higher (fixes imbalance)
      2. Group weight    → smaller age groups weighted higher (fixes fairness)

    Final weight = class_weight × group_weight per sample
    """
    # Class weights: inverse frequency
    class_counts  = y_train.value_counts()
    class_weight  = y_train.map(
        lambda c: len(y_train) / (2 * class_counts[c])
    )

    # Group weights: inverse frequency
    group_counts  = age_group_train.value_counts()
    group_weight  = age_group_train.map(
        lambda g: len(age_group_train) / (4 * group_counts[g])
    )

    sample_weights = class_weight.values * group_weight.astype(float).values
    sample_weights = sample_weights / sample_weights.mean()  # normalise
    return sample_weights


# ═══════════════════════════════════════════════════════
# STEP 4 — GET PROBABILITIES
# ═══════════════════════════════════════════════════════

def get_probabilities(pipe, X_test):
    """
    Each borrower passes through all 200 trees.
    Each tree votes 0 or 1.
    probability = votes for default / total trees

    Example:
      140 trees say default, 60 say no default
      → probability = 140/200 = 0.70
    """
    y_prob = pipe.predict_proba(X_test)[:, 1]
    print(f"\n── Probability Distribution ──")
    print(f"  Min  : {y_prob.min():.3f}")
    print(f"  Mean : {y_prob.mean():.3f}")
    print(f"  Max  : {y_prob.max():.3f}")
    return y_prob


# ═══════════════════════════════════════════════════════
# STEP 5 — EXPECTED LOSS THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════

def find_optimal_threshold(y_true, y_prob, loan_amounts, lgd=0.90):
    """
    Sweep thresholds 0.01 → 0.99 (200 values).
    At each threshold calculate total Expected Loss:

      Missed defaulter cost = loan_amount × LGD (0.90)
        → bank loses 90% of loan after failed recovery

      False alarm cost = loan_amount × 0.10
        → opportunity cost of rejecting good customer (interest lost)

    Pick threshold with minimum total loss.

    Why U-shape:
      Low threshold  → catch all defaulters but too many false alarms → high FP cost
      High threshold → miss many defaulters → high FN cost
      Middle         → optimal balance
    """
    thresholds  = np.linspace(0.01, 0.99, 200)
    total_costs = []
    fn_costs    = []
    fp_costs    = []

    for t in thresholds:
        y_pred  = (y_prob >= t).astype(int)

        fn_mask = (y_true == 1) & (y_pred == 0)   # missed defaulters
        fp_mask = (y_true == 0) & (y_pred == 1)   # false alarms

        fn_cost = (loan_amounts[fn_mask] * lgd).sum()
        fp_cost = (loan_amounts[fp_mask] * 0.10).sum()

        total_costs.append(fn_cost + fp_cost)
        fn_costs.append(fn_cost)
        fp_costs.append(fp_cost)

    best_idx  = np.argmin(total_costs)
    best_t    = thresholds[best_idx]
    best_cost = total_costs[best_idx]

    print(f"\n── Threshold Optimization ──")
    print(f"  Default threshold 0.50 loss : ₹{total_costs[100]/1e6:.2f}M")
    print(f"  Optimal threshold           : {best_t:.2f}")
    print(f"  Optimal threshold loss      : ₹{best_cost/1e6:.2f}M")
    print(f"  Saved vs default            : ₹{(total_costs[100]-best_cost)/1e6:.2f}M")

    return best_t, thresholds, total_costs, fn_costs, fp_costs


# ═══════════════════════════════════════════════════════
# STEP 6 — FAIRNESS ANALYSIS
# ═══════════════════════════════════════════════════════

def fairness_report(label, y_true, y_pred, y_prob, age_group):
    """
    For each age group measure:
      Selection Rate → % predicted as default
      TPR (Recall)   → % of actual defaulters caught
      ROC-AUC        → ranking quality within group

    TPR gap = best group TPR - worst group TPR
    Lower gap = fairer model
    """
    rows = []
    for g in sorted(age_group.unique(), key=str):
        mask = age_group == g
        yt   = y_true[mask]
        yp   = y_pred[mask]
        ypr  = y_prob[mask]
        tpr  = (yt[yp==1].sum() / yt.sum()) if yt.sum() > 0 else 0
        sr   = yp.mean()
        auc  = roc_auc_score(yt, ypr) if yt.nunique() > 1 else float("nan")
        rows.append({
            "Age Group"      : str(g),
            "Count"          : int(mask.sum()),
            "Default Rate"   : f"{yt.mean():.1%}",
            "Selection Rate" : f"{sr:.1%}",
            "TPR (Recall)"   : f"{tpr:.1%}",
            "ROC-AUC"        : f"{auc:.3f}" if not np.isnan(auc) else "N/A",
        })

    df_report = pd.DataFrame(rows)
    tpr_vals  = df_report["TPR (Recall)"].str.rstrip("%").astype(float)
    gap       = tpr_vals.max() - tpr_vals.min()

    print(f"\n── Fairness Report: {label} ──")
    print(df_report.to_string(index=False))
    print(f"\n  TPR gap : {gap:.1f}%  (best group - worst group)")
    return df_report


def per_group_threshold(y_true, y_prob, age_group, loan_amounts, lgd=0.90):
    """
    Find optimal threshold separately for each age group.

    Why this helps:
      Global threshold may be too high for elders
      → their defaulters have lower predicted probabilities
      → lower threshold for elders catches more of them
    """
    group_thresholds = {}
    print(f"\n── Per-Group Optimal Thresholds ──")

    for g in sorted(age_group.unique(), key=str):
        mask = age_group == g
        yt   = y_true[mask]
        yp   = y_prob[mask]
        la   = loan_amounts[mask]

        if yt.sum() == 0:
            group_thresholds[str(g)] = 0.5
            continue

        thresholds  = np.linspace(0.01, 0.99, 200)
        costs       = []
        for t in thresholds:
            pred    = (yp >= t).astype(int)
            fn_mask = (yt == 1) & (pred == 0)
            fp_mask = (yt == 0) & (pred == 1)
            costs.append(
                (la[fn_mask] * lgd).sum() +
                (la[fp_mask] * 0.10).sum()
            )

        best_t = thresholds[np.argmin(costs)]
        group_thresholds[str(g)] = best_t
        print(f"  {str(g):20s} → {best_t:.2f}")

    return group_thresholds


def apply_group_thresholds(y_prob, age_group, group_thresholds):
    """Apply per-group threshold to get final predictions."""
    y_pred = np.zeros(len(y_prob), dtype=int)
    for g, t in group_thresholds.items():
        mask        = age_group == g
        y_pred[mask]= (y_prob[mask] >= t).astype(int)
    return y_pred


# ═══════════════════════════════════════════════════════
# STEP 7 — VISUALIZATION
# ═══════════════════════════════════════════════════════

def plot_all(y_test, y_prob, loan_amounts_test, ag_test,
             best_t, thresholds, total_costs, fn_costs, fp_costs,
             baseline_fairness, fixed_fairness, pipe):

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Credit Risk Pipeline — Complete Results",
        fontsize=15, fontweight="bold"
    )

    # ── 1. ROC Curve ──
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"Random Forest (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(); ax.grid(alpha=0.3)

    # ── 2. Expected Loss Curve ──
    ax = axes[0, 1]
    tc = [c/1e6 for c in total_costs]
    fc = [c/1e6 for c in fn_costs]
    pc = [c/1e6 for c in fp_costs]
    ax.plot(thresholds, tc, color="#FF5722", lw=2, label="Total Loss")
    ax.plot(thresholds, fc, color="#FF9800", lw=1.5, linestyle="--", label="FN cost (missed defaults)")
    ax.plot(thresholds, pc, color="#9C27B0", lw=1.5, linestyle="--", label="FP cost (false alarms)")
    ax.axvline(best_t, color="green", linestyle="--", label=f"Optimal t={best_t:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Expected Loss (₹ millions)")
    ax.set_title("Expected Loss Optimization (U-curve)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── 3. Confusion Matrix ──
    ax = axes[0, 2]
    y_pred_opt = (y_prob >= best_t).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Default","Default"],
                yticklabels=["No Default","Default"])
    ax.set_title(f"Confusion Matrix @ threshold={best_t:.2f}")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

    # ── 4. Feature Importance ──
    ax = axes[1, 0]
    importances = pipe.named_steps["clf"].feature_importances_
    feat_names  = [
        "RevolvingUtil", "Age", "Late30-59", "DebtRatio",
        "MonthlyIncome", "OpenCredit", "Late90", "RealEstate",
        "Late60-89", "Dependents", "TotalLate"
    ]
    feat_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True)
    ax.barh(feat_df["Feature"], feat_df["Importance"], color="#2196F3", alpha=0.8)
    ax.set_title("Feature Importance"); ax.grid(alpha=0.3)

    # ── 5. Fairness: Before vs After ──
    ax = axes[1, 1]
    groups      = baseline_fairness["Age Group"]
    tpr_before  = baseline_fairness["TPR (Recall)"].str.rstrip("%").astype(float)
    tpr_after   = fixed_fairness["TPR (Recall)"].str.rstrip("%").astype(float)
    x = np.arange(len(groups)); w = 0.35
    ax.bar(x - w/2, tpr_before, w, label="Before fix", color="#FF5722", alpha=0.8)
    ax.bar(x + w/2, tpr_after,  w, label="After fix",  color="#4CAF50", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, fontsize=8)
    ax.set_ylabel("TPR / Recall (%)")
    ax.set_title("Fairness Fix: TPR by Age Group")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    for i,(b,a) in enumerate(zip(tpr_before, tpr_after)):
        ax.text(i-w/2, b+1, f"{b:.0f}%", ha="center", fontsize=7)
        ax.text(i+w/2, a+1, f"{a:.0f}%", ha="center", fontsize=7)

    # ── 6. Threshold comparison ──
    ax = axes[1, 2]
    y_pred_50   = (y_prob >= 0.50).astype(int)
    y_pred_opt  = (y_prob >= best_t).astype(int)

    def recall(yt, yp):
        tn,fp,fn,tp = confusion_matrix(yt,yp).ravel()
        return tp/(tp+fn) if (tp+fn)>0 else 0

    def el_loss(yt, yp, la):
        return (la[(yt==1)&(yp==0)] * 0.90).sum() / 1e6

    labels   = ["t=0.50\n(default)", f"t={best_t:.2f}\n(optimal)"]
    recalls  = [recall(y_test,y_pred_50)*100, recall(y_test,y_pred_opt)*100]
    losses   = [el_loss(y_test,y_pred_50,loan_amounts_test),
                el_loss(y_test,y_pred_opt,loan_amounts_test)]

    x = np.arange(2); w = 0.35
    ax2 = ax.twinx()
    ax.bar(x-w/2,  recalls, w, color="#2196F3", alpha=0.8, label="Recall %")
    ax2.bar(x+w/2, losses,  w, color="#FF5722", alpha=0.8, label="EL Loss ₹M")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Default Recall (%)"); ax2.set_ylabel("Expected Loss (₹M)")
    ax.set_title("Threshold Impact")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("outputs/complete_results.png", dpi=150, bbox_inches="tight")
    print("\n✓ Plot saved → outputs/complete_results.png")
    plt.show()


# ═══════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  COMPLETE CREDIT RISK PIPELINE")
    print("=" * 60)

    # ── Step 1: Load ──
    df = load_data()

    # ── Step 2: Preprocess ──
    X, y, age_group, loan_amt, feature_cols = preprocess(df)

    # ── Step 3: Split ──
    X_train, X_test, y_train, y_test, ag_train, ag_test, la_train, la_test = \
        train_test_split(X, y, age_group, loan_amt,
                         test_size=0.2, random_state=42, stratify=y)

    y_test   = y_test.reset_index(drop=True)
    ag_test  = ag_test.reset_index(drop=True)
    la_test  = la_test.reset_index(drop=True)
    ag_train = ag_train.reset_index(drop=True)
    y_train  = y_train.reset_index(drop=True)

    print(f"\n  Train size : {len(X_train):,}")
    print(f"  Test size  : {len(X_test):,}")

    # ── Step 4: Train ──
    print("\n[Training model with reweighting...]")
    pipe = train_model(X_train, y_train, ag_train, use_reweighting=True)

    # ── Step 5: Probabilities ──
    print("\n[Getting probabilities — 200 trees voting...]")
    y_prob = get_probabilities(pipe, X_test)
    print(f"  ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")

    # ── Step 6: Threshold optimization ──
    print("\n[Finding optimal threshold via Expected Loss sweep...]")
    best_t, thresholds, total_costs, fn_costs, fp_costs = \
        find_optimal_threshold(y_test, y_prob, la_test)

    # ── Step 7: Baseline fairness (global threshold) ──
    y_pred_global = (y_prob >= best_t).astype(int)
    baseline_fairness = fairness_report(
        f"Global threshold {best_t:.2f}",
        y_test, y_pred_global, y_prob, ag_test
    )

    # ── Step 8: Fairness fix (per-group thresholds) ──
    print("\n[Finding per-group thresholds...]")
    group_thresholds = per_group_threshold(y_test, y_prob, ag_test, la_test)
    y_pred_fair      = apply_group_thresholds(y_prob, ag_test, group_thresholds)
    fixed_fairness   = fairness_report(
        "Per-group thresholds",
        y_test, y_pred_fair, y_prob, ag_test
    )

    # ── Step 9: TPR gap summary ──
    def tpr_gap(df):
        vals = df["TPR (Recall)"].str.rstrip("%").astype(float)
        return vals.max() - vals.min()

    print(f"\n── Fairness Summary ──")
    print(f"  TPR gap before fix : {tpr_gap(baseline_fairness):.1f}%")
    print(f"  TPR gap after fix  : {tpr_gap(fixed_fairness):.1f}%")

    # ── Step 10: Final classification report ──
    print(f"\n── Final Classification Report (optimal threshold {best_t:.2f}) ──")
    print(classification_report(
        y_test, y_pred_fair,
        target_names=["No Default", "Default"]
    ))

    # ── Step 11: Plot ──
    plot_all(
        y_test, y_prob, la_test, ag_test,
        best_t, thresholds, total_costs, fn_costs, fp_costs,
        baseline_fairness, fixed_fairness, pipe
    )

    print("\n" + "=" * 60)
    print(f"  Optimal threshold : {best_t:.2f}")
    print(f"  ROC-AUC           : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  TPR gap reduced   : {tpr_gap(baseline_fairness):.1f}% → {tpr_gap(fixed_fairness):.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
