"""
train_model.py - build the synthetic dataset and train the ArthroKNEE fusion models.

Run:  python train_model.py            (about 1-2 minutes)
Makes: synthetic_arthritis_dataset.csv, arthroknee_model_v3.joblib, fig_*.png

SYNTHETIC DATA: results validate the pipeline only, not clinical performance.
"""
import sys
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arthro_core as core

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
rng = np.random.default_rng(42)
print(f"Generating {N} synthetic patients ...")
syn = core.generate(N, rng)
syn.to_csv("synthetic_arthritis_dataset.csv", index=False)
print(" condition mix:", {core.COND_NAMES[k]: int(v) for k, v in syn['condition'].value_counts().sort_index().items()})
print(" risk mix     :", {core.RISK_NAMES[k]: int(v) for k, v in syn['risk'].value_counts().sort_index().items()})

print("Training 3 fusion models + ablation study ...")
bundle = core.train_all(syn, ablation=True)
joblib.dump(bundle, "arthroknee_model_v3.joblib", compress=3)
m = bundle["metrics"]

print("\n=== Hold-out results (25% of synthetic data, never used for training) ===")
print(f" Stage 1  arthritis screening : accuracy {m['screen_acc']:.3f} | AUC {m['screen_auc']:.3f} | "
      f"sensitivity {m['sensitivity']:.3f} | specificity {m['specificity']:.3f}")
print(f" Stage 2  OA vs RA            : accuracy {m['type_acc']:.3f} | AUC {m['type_auc']:.3f}")
print(f" 3-way (none / OA / RA)       : accuracy {m['cond_acc']:.3f}")
print(f" Risk level (low/mod/high)    : accuracy {m['risk_acc']:.3f}")
print("\n Why fusion? (same test set)")
for k, v in m["ablation"].items():
    print(f"   {k:20s} screening AUC {v['screen_auc']:.3f} | 3-way accuracy {v['cond_acc']:.3f}")
print("\n Share of model importance by modality:",
      {k: f"{100*v:.0f}%" for k, v in m["importance_modality"].items()})

# ---- figures for the slides
names = list(m["ablation"])
fig, ax = plt.subplots(figsize=(7.5, 4))
x = np.arange(len(names)); wd = 0.38
ax.bar(x - wd / 2, [m["ablation"][n]["screen_auc"] for n in names], wd, label="Screening AUC", color="#1d6f8c")
ax.bar(x + wd / 2, [m["ablation"][n]["cond_acc"] for n in names], wd, label="3-way accuracy", color="#e8a317")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right"); ax.set_ylim(0, 1)
ax.set_title("Single modality vs multimodal fusion (synthetic data)"); ax.legend()
plt.tight_layout(); plt.savefig("fig_fusion_ablation.png", dpi=150); plt.close()

cm = np.array(m["cond_confusion"])
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(core.COND_NAMES); ax.set_yticklabels(core.COND_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("None / OA / RA (hold-out, synthetic)")
plt.tight_layout(); plt.savefig("fig_confusion_condition.png", dpi=150); plt.close()

imp = sorted(m["importance_features"].items(), key=lambda kv: kv[1])[-15:]
fig, ax = plt.subplots(figsize=(6.5, 5))
cols = {"quest": "#7b61c4", "wear": "#1d6f8c", "gait": "#e8a317"}
ax.barh([k for k, _ in imp], [v for _, v in imp], color=[cols[core.SPEC[k]["modality"]] for k, _ in imp])
ax.set_title("Top 15 features (purple = questionnaire, teal = wearable, gold = gait)")
plt.tight_layout(); plt.savefig("fig_feature_importance.png", dpi=150); plt.close()
print("\nSaved: arthroknee_model_v3.joblib, synthetic_arthritis_dataset.csv, fig_*.png")
