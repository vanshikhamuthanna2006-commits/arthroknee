"""
arthro_core.py - ArthroKNEE core (SYNTHETIC-DATA PROOF OF CONCEPT)

* FEATS        : the 42 inputs, tagged by modality (questionnaire / wearable / gait camera)
* generate()   : realistic synthetic patients (healthy / OA / RA) with overlap and noise
* train_all()  : 3 fusion models  ->  Stage 1 arthritis?  Stage 2 OA vs RA  ->  risk level
* assess()     : prediction for one patient;  explain(): occlusion-based explanation
* checklist()  : simple clinical-pattern cross-check (inspired by ACR / EULAR features)

The synthetic generator encodes literature-style tendencies (OA: older, higher BMI, short
morning stiffness, crepitus, asymmetric; RA: younger, female, long morning stiffness,
symmetric small-joint involvement, warmth, raised CRP/ESR, RF/anti-CCP). It is NOT real
patient data: accuracy on it validates the PIPELINE, not clinical performance.
"""
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# key, modality, group, kind, lo, hi, step, default, scale
FEATS = [
    # ---- questionnaire & clinical history
    ("age", "quest", "history", "int", 18, 90, 1, 45, None),
    ("sex_female", "quest", "history", "binary", 0, 1, 1, 1, None),
    ("bmi", "quest", "history", "float", 15.0, 50.0, 0.5, 24.0, None),
    ("prior_injury", "quest", "history", "binary", 0, 1, 1, 0, None),
    ("family_autoimmune", "quest", "history", "binary", 0, 1, 1, 0, None),
    ("smoker", "quest", "history", "binary", 0, 1, 1, 0, None),
    ("pain_rest", "quest", "pain", "int", 0, 10, 1, 1, None),
    ("pain_walk", "quest", "pain", "int", 0, 10, 1, 1, None),
    ("pain_stairs", "quest", "pain", "int", 0, 10, 1, 1, None),
    ("pain_night", "quest", "pain", "int", 0, 10, 1, 0, None),
    ("diff_chair", "quest", "function", "likert", 0, 4, 1, 0, "sev"),
    ("diff_stairs", "quest", "function", "likert", 0, 4, 1, 0, "sev"),
    ("diff_shoes", "quest", "function", "likert", 0, 4, 1, 0, "sev"),
    ("morning_stiff_min", "quest", "stiff", "int", 0, 240, 5, 5, None),
    ("gelling", "quest", "stiff", "likert", 0, 4, 1, 0, "sev"),
    ("mech_pattern", "quest", "stiff", "likert", 0, 4, 1, 1, "frq"),
    ("infl_pattern", "quest", "stiff", "likert", 0, 4, 1, 0, "frq"),
    ("small_joints", "quest", "joints", "int", 0, 10, 1, 0, None),
    ("symmetry", "quest", "joints", "likert", 0, 4, 1, 0, "frq"),
    ("swelling_warmth", "quest", "joints", "likert", 0, 4, 1, 0, "sev"),
    ("fatigue", "quest", "joints", "int", 0, 10, 1, 1, None),
    ("systemic", "quest", "joints", "likert", 0, 4, 1, 0, "sev"),
    ("duration_months", "quest", "joints", "int", 0, 120, 1, 3, None),
    ("grinding", "quest", "joints", "likert", 0, 4, 1, 0, "frq"),
    ("giving_way", "quest", "joints", "likert", 0, 4, 1, 0, "frq"),
    ("bony_change", "quest", "joints", "likert", 0, 4, 1, 0, "sev"),
    ("labs_known", "quest", "labs", "binary", 0, 1, 1, 0, None),
    ("crp_mgL", "quest", "labs", "float", 0.0, 150.0, 0.5, 3.0, None),
    ("esr_mmhr", "quest", "labs", "int", 0, 120, 1, 15, None),
    ("rf_pos", "quest", "labs", "binary", 0, 1, 1, 0, None),
    ("ccp_pos", "quest", "labs", "binary", 0, 1, 1, 0, None),
    # ---- wearable sensors (ESP32: flex, 2x MPU6050, temperature, strain)
    ("flexion_rom_deg", "wear", "sensor", "float", 60.0, 150.0, 1.0, 135.0, None),
    ("rom_asym_pct", "wear", "sensor", "float", 0.0, 35.0, 0.5, 4.0, None),
    ("imu_sway_rms", "wear", "sensor", "float", 0.0, 3.0, 0.05, 0.5, None),
    ("temp_delta_C", "wear", "sensor", "float", 0.0, 3.0, 0.05, 0.2, None),
    ("swelling_idx", "wear", "sensor", "float", 0.0, 100.0, 1.0, 10.0, None),
    ("sts_time_s", "wear", "sensor", "float", 5.0, 30.0, 0.5, 10.0, None),
    ("crepitus_idx", "wear", "sensor", "float", 0.0, 100.0, 1.0, 10.0, None),
    # ---- gait (camera)
    ("gait_knee_peak_deg", "gait", "gait", "float", 30.0, 75.0, 1.0, 64.0, None),
    ("stride_cv_pct", "gait", "gait", "float", 0.0, 12.0, 0.1, 2.5, None),
    ("step_asym_pct", "gait", "gait", "float", 0.0, 25.0, 0.5, 3.0, None),
    ("cadence_spm", "gait", "gait", "float", 50.0, 140.0, 1.0, 112.0, None),
]
FEATURES = [f[0] for f in FEATS]
SPEC = {f[0]: dict(modality=f[1], group=f[2], kind=f[3], lo=f[4], hi=f[5],
                   step=f[6], default=f[7], scale=f[8]) for f in FEATS}
MODALITIES = {"quest": [f for f in FEATURES if SPEC[f]["modality"] == "quest"],
              "wear": [f for f in FEATURES if SPEC[f]["modality"] == "wear"],
              "gait": [f for f in FEATURES if SPEC[f]["modality"] == "gait"]}
COND_NAMES = ["No arthritis", "OA", "RA"]
RISK_NAMES = ["Low", "Moderate", "High"]


def cast(key, v):
    """Clip a value to the feature's range and give it the type the widget expects."""
    s = SPEC[key]
    v = float(np.clip(v, s["lo"], s["hi"]))
    if s["kind"] in ("int", "likert", "binary"):
        return int(round(v))
    step = s["step"]
    return float(round(round(v / step) * step, 4))


def default_values():
    return {k: cast(k, SPEC[k]["default"]) for k in FEATURES}


# ===================================================================== generator
def generate(n, rng, cond=None, sev=None):
    """n synthetic patients. cond: None (mixed) or 0/1/2. sev: None or severity 0-1.
    Realism features: many mild/early cases overlapping with healthy people who have everyday
    aches; atypical phenotypes (each modality may show an OA-like or RA-like picture
    independently); person-level reporting/sensor offsets; label noise on risk."""
    cond = rng.choice([0, 1, 2], size=n, p=[0.42, 0.34, 0.24]) if cond is None \
        else np.full(n, cond)
    oa, ra = (cond == 1).astype(float), (cond == 2).astype(float)
    arth = cond > 0
    s = np.where(arth, rng.beta(1.2, 2.5, n) if sev is None else sev, 0.0)
    w = np.where(arth, 0.0, rng.beta(1.5, 3.2, n) * 0.7)        # everyday aches, no arthritis
    if sev is not None and np.all(cond == 0):               # virtual healthy: severity = level of everyday aches
        w = sev * 0.4 * rng.uniform(0.5, 1.0, n)
    eff = s + w
    N = lambda sd: rng.normal(0, sd, n)
    # each modality sees the disease through its own person-level offset (independent noise)
    eq = np.clip(eff + N(0.22), 0, 1.05)   # self-report: subjective, includes everyday aches
    ew = np.clip(s + 0.25 * w + N(0.14), 0, 1.05)   # sensors: objective pathology, little response to mere aches
    eg = np.clip(s + 0.40 * w + N(0.18), 0, 1.05)

    style = np.exp(rng.normal(0, 0.40, n))           # reporting style: stoic <-> expressive
    eqs = np.clip(eq * style, 0, 1.3)

    def mixer():
        """phenotype blend: 0 = classic picture, larger = atypical (OA looks RA-like or vice versa)"""
        m = rng.beta(1.2, 4.0, n)
        ra_w = np.where(ra == 1, 1 - m, np.where(oa == 1, m, 0.5))
        return lambda o, r: o * (1 - ra_w) + r * ra_w
    fq, fw, fg = mixer(), mixer(), mixer()

    d = {}
    d["age"] = np.where(cond == 1, rng.normal(57, 11, n),
                        np.where(cond == 2, rng.normal(47, 13, n), rng.normal(45, 15, n)))
    d["sex_female"] = rng.random(n) < np.where(cond == 2, 0.70, np.where(cond == 1, 0.57, 0.5))
    d["bmi"] = np.where(cond == 1, rng.normal(28.5, 4.8, n),
                        np.where(cond == 2, rng.normal(26, 4.8, n), rng.normal(25.5, 4.5, n)))
    d["prior_injury"] = rng.random(n) < np.where(cond == 1, 0.30, np.where(cond == 2, 0.08, 0.11))
    d["family_autoimmune"] = rng.random(n) < np.where(cond == 2, 0.27, 0.09)
    d["smoker"] = rng.random(n) < np.where(cond == 2, 0.33, 0.20)
    age, bmi = d["age"], d["bmi"]

    L = lambda b, sl, f=1.0, sd=0.8: np.rint(b + sl * eqs * f + N(sd))          # 0-4 items
    P = lambda b, sl, f=1.0, sd=1.5: np.rint(b + sl * eqs * f + N(sd))          # 0-10 items
    d["pain_rest"] = P(0.4, 5.0, fq(0.55, 1.0), 1.3)
    d["pain_walk"] = P(0.7, 6.8, 1.0, 1.7)
    d["pain_stairs"] = P(0.9, 7.0, 1.0, 1.7)
    d["pain_night"] = P(0.3, 5.5, fq(0.5, 1.0), 1.4)
    d["diff_chair"] = L(0.25, 3.2)
    d["diff_stairs"] = L(0.25, 3.3)
    d["diff_shoes"] = L(0.15, 2.9, fq(0.6, 1.0))
    ms = np.where(arth, 4 + eqs * fq(22, 110), 4 + 26 * eqs)
    d["morning_stiff_min"] = np.rint(ms * np.exp(N(0.5)) / 5) * 5
    d["gelling"] = L(0.35, 3.0, fq(1.0, 0.75))
    d["mech_pattern"] = L(0.6, 3.0, fq(1.0, 0.45))
    d["infl_pattern"] = L(0.35, 3.0, fq(0.3, 1.0))
    lam = np.where(arth, 0.35 + eqs * fq(1.8, 6.0), 0.3 + 1.0 * eqs)
    d["small_joints"] = rng.poisson(lam)
    d["symmetry"] = L(0.5, 3.0, fq(0.4, 1.0))
    d["swelling_warmth"] = L(0.25, 3.1, fq(0.42, 1.0))
    d["fatigue"] = P(1.2, 5.6, fq(0.45, 1.0), 1.7)
    d["systemic"] = L(0.15, 2.1, fq(0.22, 1.0))
    med = np.where(ra == 1, 10, np.where(oa == 1, 20, 8))
    d["duration_months"] = np.rint(rng.lognormal(np.log(med), 0.95, n))
    d["grinding"] = L(0.25, 3.1, fq(1.0, 0.4))
    d["giving_way"] = L(0.15, 2.3, fq(1.0, 0.6))
    d["bony_change"] = L(0.15, 2.6, fq(1.0, 0.28))

    crp = np.where(arth, 2.5 + eq * fq(4, 36), 2.2 + 4.0 * eq) * np.exp(N(0.65))
    esr = (np.where(arth, 12 + eq * fq(9, 40), 10 + 9 * eq) * np.exp(N(0.35)) + 0.15 * (age - 40))
    rf = rng.random(n) < np.where(ra == 1, 0.45 + 0.25 * eq, np.where(oa == 1, 0.09, 0.06))
    ccp = rng.random(n) < np.where(ra == 1, 0.35 + 0.30 * eq, 0.03)
    known = rng.random(n) < 0.5
    d["labs_known"] = known
    d["crp_mgL"] = np.where(known, crp, 3.0)
    d["esr_mmhr"] = np.where(known, esr, 15.0)
    d["rf_pos"] = np.where(known, rf, 0)
    d["ccp_pos"] = np.where(known, ccp, 0)

    d["flexion_rom_deg"] = 138 - 48 * ew - 6 * oa * ew - 0.12 * (age - 40) + N(8)
    d["rom_asym_pct"] = 3 + 19 * ew * fw(1.0, 0.55) + N(4.2)
    d["imu_sway_rms"] = 0.5 + 1.45 * ew + 0.004 * (age - 40) + N(0.36)
    d["temp_delta_C"] = 0.15 + 1.6 * ew * fw(0.3, 1.0) + N(0.38)
    d["swelling_idx"] = 8 + 58 * ew * fw(0.5, 1.0) + N(10)
    d["sts_time_s"] = 9 + 8.5 * ew + 0.04 * (age - 40) + 0.08 * (bmi - 25) + N(2.0)
    d["crepitus_idx"] = 10 + 52 * ew * fw(1.0, 0.3) + 0.25 * (age - 40) + N(11)

    d["gait_knee_peak_deg"] = 64 - 20 * eg + N(5.5)
    d["stride_cv_pct"] = 2.2 + 5.5 * eg + N(1.3)
    d["step_asym_pct"] = 3 + 12 * eg * fg(1.0, 0.65) + N(3.4)
    d["cadence_spm"] = 112 - 26 * eg + N(8)

    df = pd.DataFrame({k: d[k] for k in FEATURES})
    for k in FEATURES:                                    # clip + type
        sp = SPEC[k]
        col = np.clip(df[k].astype(float), sp["lo"], sp["hi"])
        df[k] = col.round().astype(int) if sp["kind"] in ("int", "likert", "binary") \
            else col.round(2)

    # risk label: disease severity (or at-risk non-arthritis with aches), plus 4% label noise
    at_risk = (~arth) & (w > 0.25) & ((bmi > 30) | d["family_autoimmune"] | ((age > 55) & d["smoker"]))
    risk = np.where(arth, np.where(s < 0.40, 1, 2), np.where(at_risk, 1, 0))
    flip = rng.random(n) < 0.04
    risk = np.clip(risk + np.where(flip, rng.choice([-1, 1], n), 0), 0, 2)
    df["condition"], df["risk"], df["severity"] = cond, risk, np.round(s, 3)
    return df


# ===================================================================== training
def _rf(n=300, seed=0):
    return RandomForestClassifier(n_estimators=n, min_samples_leaf=8, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=seed)


def _proba(model, X, cls):
    """probability of class `cls` (robust to missing classes)."""
    p = model.predict_proba(X)
    return p[:, list(model.classes_).index(cls)] if cls in model.classes_ else np.zeros(len(X))


def _cond_probs(m_screen, m_type, X):
    pa = _proba(m_screen, X, 1)
    pr = _proba(m_type, X, 1)
    return np.c_[1 - pa, pa * (1 - pr), pa * pr]


def train_all(syn, real=None, real_weight=8.0, ablation=True, seed=0, n_trees=200):
    """syn: synthetic df (FEATURES + condition, risk). real: confirmed real cases (same columns)."""
    tr, te = train_test_split(syn, test_size=0.25, stratify=syn["condition"], random_state=seed)
    parts, weights = [tr], [np.ones(len(tr))]
    if real is not None and len(real):
        parts.append(real)
        weights.append(np.full(len(real), real_weight))
    trn = pd.concat(parts, ignore_index=True)
    w = np.concatenate(weights)

    def fit_stage1(cols, n):
        return _rf(n, seed).fit(trn[cols], (trn["condition"] > 0).astype(int), sample_weight=w)

    def fit_stage2(cols, n):
        m = (trn["condition"] > 0).values
        return _rf(n, seed).fit(trn.loc[m, cols], (trn.loc[m, "condition"] == 2).astype(int),
                                sample_weight=w[m])

    m_screen, m_type = fit_stage1(FEATURES, n_trees), fit_stage2(FEATURES, n_trees)
    rk = trn["risk"].notna().values
    m_risk = _rf(n_trees, seed).fit(trn.loc[rk, FEATURES], trn.loc[rk, "risk"].astype(int),
                                    sample_weight=w[rk])

    Xte = te[FEATURES]
    cp = _cond_probs(m_screen, m_type, Xte)
    cond_pred = cp.argmax(1)
    pa = cp[:, 1] + cp[:, 2]
    y_arth = (te["condition"] > 0).astype(int)
    pred_arth = (pa >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arth, pred_arth).ravel()
    am = (te["condition"] > 0).values
    ra_p = cp[am, 2] / np.maximum(cp[am, 1] + cp[am, 2], 1e-9)
    y_ra = (te.loc[am, "condition"] == 2).astype(int)
    risk_pred = m_risk.predict(Xte)
    metrics = dict(
        n_test=int(len(te)),
        screen_auc=float(roc_auc_score(y_arth, pa)),
        screen_acc=float(accuracy_score(y_arth, pred_arth)),
        sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
        type_auc=float(roc_auc_score(y_ra, ra_p)),
        type_acc=float(accuracy_score(y_ra, (ra_p >= 0.5).astype(int))),
        cond_acc=float(accuracy_score(te["condition"], cond_pred)),
        risk_acc=float(accuracy_score(te["risk"], risk_pred)),
        cond_confusion=confusion_matrix(te["condition"], cond_pred, labels=[0, 1, 2]).tolist(),
        risk_confusion=confusion_matrix(te["risk"], risk_pred, labels=[0, 1, 2]).tolist(),
    )
    imp = pd.Series(m_screen.feature_importances_ + m_type.feature_importances_, index=FEATURES)
    imp = imp / imp.sum()
    metrics["importance_features"] = imp.to_dict()
    metrics["importance_modality"] = {m: float(imp[c].sum()) for m, c in MODALITIES.items()}

    if ablation:
        subsets = {"Questionnaire only": MODALITIES["quest"], "Wearable only": MODALITIES["wear"],
                   "Gait camera only": MODALITIES["gait"],
                   "Wearable + Gait": MODALITIES["wear"] + MODALITIES["gait"],
                   "FUSION (all 3)": FEATURES}
        abl = {}
        for name, cols in subsets.items():
            s1, s2 = fit_stage1(cols, 150), fit_stage2(cols, 150)
            pa_ = _proba(s1, te[cols], 1)
            pr_ = _proba(s2, te[cols], 1)
            cond_ = np.c_[1 - pa_, pa_ * (1 - pr_), pa_ * pr_].argmax(1)
            abl[name] = dict(screen_auc=float(roc_auc_score(y_arth, pa_)),
                             cond_acc=float(accuracy_score(te["condition"], cond_)))
        metrics["ablation"] = abl

    ref = syn.loc[syn["condition"] == 0, FEATURES].median().to_dict()
    for m in (m_screen, m_type, m_risk):
        m.set_params(n_jobs=1)                           # light-weight inference
    return dict(models=dict(screen=m_screen, type=m_type, risk=m_risk), features=FEATURES,
                ref={k: cast(k, v) for k, v in ref.items()}, metrics=metrics,
                meta=dict(n_synthetic=int(len(syn)), n_real=int(0 if real is None else len(real)),
                          trained_at=datetime.datetime.now().isoformat(timespec="seconds")))


# ===================================================================== inference
def assess(bundle, values):
    ms, mt, mr = (bundle["models"][k] for k in ("screen", "type", "risk"))
    X = pd.DataFrame([values])[FEATURES]
    pa = float(_proba(ms, X, 1)[0])
    pr = float(_proba(mt, X, 1)[0])
    rp = np.array([float(_proba(mr, X, c)[0]) for c in (0, 1, 2)])
    rp = rp / max(rp.sum(), 1e-9)
    return dict(p_arth=pa, p_ra_given=pr, p_none=1 - pa, p_oa=pa * (1 - pr), p_ra=pa * pr,
                risk_p=rp.tolist(), risk=int(rp.argmax()),
                risk_score=float(100 * (0.5 * rp[1] + rp[2])))


def explain(bundle, values):
    """Occlusion explanation: replace a feature (or a whole modality) by the healthy reference
    value and see how much the arthritis probability / RA-lean changes."""
    ms, mt = bundle["models"]["screen"], bundle["models"]["type"]
    ref = bundle["ref"]
    rows = [dict(values)]
    for f in FEATURES:
        r = dict(values); r[f] = ref[f]; rows.append(r)
    for m in ("quest", "wear", "gait"):
        r = dict(values)
        for f in MODALITIES[m]:
            r[f] = ref[f]
        rows.append(r)
    X = pd.DataFrame(rows)[FEATURES]
    pa, pr = _proba(ms, X, 1), _proba(mt, X, 1)
    d_arth = pa[0] - pa[1:1 + len(FEATURES)]
    d_ra = pr[0] - pr[1:1 + len(FEATURES)]
    mod = {m: float(pa[0] - pa[1 + len(FEATURES) + i]) for i, m in enumerate(("quest", "wear", "gait"))}
    feat = pd.DataFrame({"feature": FEATURES, "d_arth": d_arth, "d_ra": d_ra})
    return dict(modality=mod, features=feat)


def checklist(v):
    """Rule-based clinical-pattern cross-check (simplified, inspired by ACR / EULAR features)."""
    labs = bool(v["labs_known"])
    oa = [("Age 50 or older", v["age"] >= 50),
          ("Morning stiffness 30 min or less", v["morning_stiff_min"] <= 30),
          ("Grinding / crackling felt often", v["grinding"] >= 2),
          ("Bony enlargement at the joint line", v["bony_change"] >= 2),
          ("Little warmth or swelling", v["swelling_warmth"] <= 1),
          ("Pain worse with activity, better with rest", v["mech_pattern"] >= 2),
          ("One-sided / not symmetric", v["symmetry"] <= 1)]
    ra = [("Morning stiffness over 60 min", v["morning_stiff_min"] > 60),
          ("3 or more small joints involved", v["small_joints"] >= 3),
          ("Symmetric joint involvement", v["symmetry"] >= 3),
          ("Visible swelling / warmth", v["swelling_warmth"] >= 2),
          ("Stiffness eases with movement", v["infl_pattern"] >= 2),
          ("Symptoms for 6+ weeks", v["duration_months"] >= 2),
          ("Raised CRP or ESR" + ("" if labs else " (labs not entered)"),
           labs and (v["crp_mgL"] > 10 or v["esr_mmhr"] > 30)),
          ("RF or anti-CCP positive" + ("" if labs else " (labs not entered)"),
           labs and (bool(v["rf_pos"]) or bool(v["ccp_pos"])))]
    return oa, ra


def questionnaire_scores(v):
    """Simple 0-100 subscale summaries (higher = worse)."""
    pain = np.mean([v["pain_rest"], v["pain_walk"], v["pain_stairs"], v["pain_night"]]) * 10
    func = np.mean([v["diff_chair"], v["diff_stairs"], v["diff_shoes"]]) * 25
    stiff = 50 * min(v["morning_stiff_min"] / 120, 1) + 12.5 * (v["gelling"])
    return dict(pain=float(pain), function=float(func), stiffness=float(min(stiff, 100)))
