"""Credit-card fraud on the European cardholder dataset: imbalance, PR curves, cost."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data/creditcard.csv"))
y = df["Class"]
X = df.drop(columns=["Class"])
X["Amount_scaled"] = StandardScaler().fit_transform(df[["Amount"]])
X = X.drop(columns=["Amount"])
tr, te = train_test_split(X.index.values, test_size=0.25, stratify=y, random_state=42)
X_tr, X_te, y_tr, y_te = X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]

lr = LogisticRegression(max_iter=5000, class_weight="balanced")
gb = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
iso = IsolationForest(n_estimators=200, random_state=42, contamination=0.002)
lr.fit(X_tr, y_tr); gb.fit(X_tr, y_tr); iso.fit(X_tr)
s = -iso.score_samples(X_te)
res = {"fraud_rate": round(float(y.mean()), 5),
       "lr_pr_auc": round(float(average_precision_score(y_te, lr.predict_proba(X_te)[:, 1])), 4),
       "gb_pr_auc": round(float(average_precision_score(y_te, gb.predict_proba(X_te)[:, 1])), 4),
       "iso_pr_auc": round(float(average_precision_score(y_te, s)), 4),
       "gb_roc_auc": round(float(roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1])), 5)}

p = gb.predict_proba(X_te)[:, 1]
prec, rec, th = precision_recall_curve(y_te, p)
ok = th <= 0.20
res["recall_at_threshold_0_20"] = round(float(rec[:-1][ok].max()), 4)
res["precision_at_threshold_0_20"] = round(float(prec[:-1][ok].max()), 4)
flag = (p >= 0.20)
FRAUD_LOSS, REVIEW_COST = 120.0, 4.0
saved = float(((y_te == 1) & flag).sum() * FRAUD_LOSS)
spent = float((flag & (y_te == 0)).sum() * REVIEW_COST)
res["net_saving_per_month_sim"] = round(saved - spent, 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for name, sc_ in [("gradient boosting", p), ("logistic", lr.predict_proba(X_te)[:, 1]), ("isolation forest", s)]:
    pr_, rc_, _ = precision_recall_curve(y_te, sc_)
    axes[0].plot(rc_, pr_, label=name)
axes[0].axhline(float(y_te.mean()), ls="--", lw=.8, color="k", label="no-skill")
axes[0].legend(fontsize=7); axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision"); axes[0].set_title("Precision-recall")
axes[1].bar(["caught fraud (x$120)", "false alarms (x$4)"], [((y_te == 1) & flag).sum(), (flag & (y_te == 0)).sum()],
            color=["tab:red", "tab:gray"])
axes[1].set_yscale("log"); axes[1].set_title("Detections at threshold 0.20 (log)")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures/fraud.png"), dpi=110); plt.close()

metrics = {"results": res,
           "table": {"title": "Operating point (threshold 0.20)",
                     "headers": ["measure", "value"],
                     "rows": [["fraud caught", int(((y_te == 1) & flag).sum())],
                              ["fraud missed", int(((y_te == 1) & ~flag).sum())],
                              ["false alarms", int((flag & (y_te == 0)).sum())],
                              ["alert rate", round(float(flag.mean()), 5)],
                              ["net saving in sim window", saved - spent]]},
           "figures": [["fraud.png", "PR curves and the detection trade-off at one threshold"]]}
json.dump(metrics, open(os.path.join(ROOT, "metrics.json"), "w"), indent=1)
print(json.dumps(res))
