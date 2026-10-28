# -*- coding: utf-8 -*-
"""
ArthroKNEE v3 - Early arthritis screening by multimodal fusion (SYNTHETIC-DATA PROOF OF CONCEPT)

  Questionnaire + Wearable sensors + Camera gait  ->  3 fusion models
  Stage 1: arthritis likely?   Stage 2: OA or RA?   Risk level: Low / Moderate / High

Run:   python -m streamlit run arthroknee_app.py
Needs: opencv-python-headless, scipy (see requirements.txt). The model trains itself on first launch.
Files: arthro_core.py, translations.py, knee_vision.py in the same folder.
"""
import os
import html
import hashlib
import tempfile
import datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st

import arthro_core as core
from translations import TR, LANGS

try:
    import knee_vision as kv
    CV_OK = True
except Exception:
    CV_OK = False

st.set_page_config(page_title="ArthroKNEE", page_icon="🦵", layout="wide")

# True when hosted online (Streamlit Community Cloud): heavy / data-changing features are switched off
DEPLOYED = os.path.exists("/mount/src") or os.environ.get("ARTHRO_ONLINE") == "1"

# ================================================================== language
st.sidebar.markdown("### 🦵 ArthroKNEE")
lang_label = st.sidebar.selectbox("🌐 Language / भाषा", list(LANGS), key="lang_label")
LANG = LANGS[lang_label]


def tr(key, **kw):
    s = TR.get(LANG, {}).get(key) or TR["en"].get(key, key)
    return s.format(**kw) if kw else s


# ================================================================== styling
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg,#0b3c5d 0%,#1d6f8c 55%,#3aa6a0 100%); color:#fff;
       border-radius: 18px; padding: 20px 26px; margin-bottom: 14px;
       box-shadow: 0 8px 24px rgba(11,60,93,.25);}
.hero h2 {margin: 0; color:#fff; font-weight: 800; letter-spacing: .3px;}
.hero .sub {opacity:.92; font-size:.95rem; margin-top:2px;}
.pill {display:inline-block; background: rgba(255,255,255,.18); border-radius: 999px;
       padding: 3px 12px; margin: 8px 6px 0 0; font-size:.82rem;}
.card {background: rgba(128,128,128,0.07); border: 1px solid rgba(128,128,128,0.22);
       border-radius: 16px; padding: 16px 20px; margin-bottom: 12px;}
.mod {border-radius: 16px; padding: 14px 18px; color:#fff; min-height: 96px;
      box-shadow: 0 4px 14px rgba(0,0,0,.12);}
.mod .t {font-size:.85rem; opacity:.92;} .mod .v {font-size:1.7rem; font-weight:800; line-height:1.2;}
.mod .s {font-size:.78rem; opacity:.85;}
.m-q {background: linear-gradient(135deg,#6a4fc2,#9a84e6);}
.m-w {background: linear-gradient(135deg,#0f7b8a,#35b3a8);}
.m-g {background: linear-gradient(135deg,#d98b0a,#f0b84a);}
.verdict {border-radius: 16px; padding: 14px 20px; color:#fff; min-height: 96px;
          box-shadow: 0 4px 14px rgba(0,0,0,.15);}
.verdict .t {font-size:.85rem; opacity:.92;} .verdict .v {font-size:1.25rem; font-weight:800; line-height:1.25;}
.badge {display:inline-block; padding:6px 18px; border-radius:999px; color:#fff;
        font-weight:800; font-size:1.15rem; letter-spacing:.5px;}
.big {font-size: 2.6rem; font-weight: 800; line-height: 1.1;}
.small {font-size:.85rem; opacity:.75;}
.gauge {position:relative; height:16px; border-radius:8px; background: rgba(128,128,128,.25); margin:10px 0 4px;}
.gauge .fill {height:100%; border-radius:8px;}
.gauge .tick {position:absolute; left:50%; top:-4px; bottom:-4px; width:2px; background:#888;}
.split {display:flex; height:38px; border-radius:10px; overflow:hidden; margin:8px 0;}
.split div {display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700;
            font-size:.92rem; min-width: 3%; white-space:nowrap;}
.bar {height:11px; border-radius:6px; background: rgba(128,128,128,.25); overflow:hidden; margin:3px 0 9px;}
.bar > div {height:100%; border-radius:6px;}
.row {display:flex; justify-content:space-between; font-size:.9rem; margin-top:4px;}
.chk {padding: 3px 0; font-size:.92rem;}
.drv {padding: 5px 0; border-bottom: 1px dashed rgba(128,128,128,.3); font-size:.9rem;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
.stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 8px 14px;}
</style>
""", unsafe_allow_html=True)

# ================================================================== model + storage
MODEL_FILE = "arthroknee_model_v3.joblib"
SYN_FILE = "synthetic_arthritis_dataset.csv"
PAT_FILE, SES_FILE, REAL_FILE = "ak_patients.csv", "ak_sessions.csv", "ak_real_data.csv"
FEATURES = core.FEATURES
COLORS = {0: "#2e9e5b", 1: "#e8a317", 2: "#d64545"}
C_OA, C_RA = "#5b6cdb", "#e0605a"
LVL_KEY = {0: "lvl_low", 1: "lvl_mod", 2: "lvl_high"}

SENSOR_LABEL = {
    "flexion_rom_deg": "Knee flexion ROM (deg) - flex sensor + IMU",
    "rom_asym_pct": "ROM asymmetry between knees (%)",
    "imu_sway_rms": "Instability / sway (IMU RMS)",
    "temp_delta_C": "Knee skin temperature rise (°C)",
    "swelling_idx": "Swelling index (strain sensor, 0-100)",
    "sts_time_s": "5x sit-to-stand time (s)",
    "crepitus_idx": "Joint crepitus / vibration index (0-100)",
    "gait_knee_peak_deg": "Peak knee flexion in gait (deg)",
    "stride_cv_pct": "Stride-time variability (%)",
    "step_asym_pct": "Step asymmetry (%)",
    "cadence_spm": "Cadence (steps/min)",
}


def label(k):
    return tr(f"q_{k}") if f"q_{k}" in TR["en"] else SENSOR_LABEL.get(k, k)


# --- robust input coercion: whatever a widget stores (number, bool, or a text label in ANY language)
def _reverse_labels():
    rev = {"sev": {}, "frq": {}, "yn": {}, "sex": {}}
    for d in TR.values():
        for sc in ("sev", "frq"):
            for v in range(5):
                rev[sc][(d.get(f"{sc}_{v}") or TR["en"][f"{sc}_{v}"]).strip().lower()] = v
        rev["yn"][(d.get("yes") or "Yes").strip().lower()] = 1
        rev["yn"][(d.get("no") or "No").strip().lower()] = 0
        rev["sex"][(d.get("female") or "Female").strip().lower()] = 1
        rev["sex"][(d.get("male") or "Male").strip().lower()] = 0
    return rev


REV = _reverse_labels()


def coerce(k, v):
    """Return a clean number for feature k, even if v is a label like 'Male', 'Mild' or 'हाँ'."""
    sp = core.SPEC[k]
    if isinstance(v, str):
        t = v.strip().lower()
        if k == "sex_female":
            v = REV["sex"].get(t)
        elif sp["kind"] == "binary":
            v = REV["yn"].get(t)
        elif sp["kind"] == "likert":
            v = REV[sp["scale"]].get(t)
        else:
            try:
                v = float(t)
            except ValueError:
                v = None
    if v is None or (isinstance(v, float) and np.isnan(v)):
        v = sp["default"]
    return core.cast(k, v)


def build_model():
    """Train the models from scratch (used on first launch, or if the saved file can't be read)."""
    syn = pd.read_csv(SYN_FILE) if os.path.exists(SYN_FILE) else core.generate(12000, np.random.default_rng(42))
    nb = core.train_all(syn, ablation=not DEPLOYED, n_trees=100 if DEPLOYED else 200)   # lean on small servers
    joblib.dump(nb, MODEL_FILE, compress=3)
    return nb


@st.cache_resource(show_spinner="First-time setup: training the models (about 1-2 minutes, once only) ...")
def load_bundle(version):
    try:
        return joblib.load(MODEL_FILE)            # fast path
    except Exception:                              # missing file, or made by another scikit-learn version
        return build_model()


MODEL_VER = os.path.getmtime(MODEL_FILE) if os.path.exists(MODEL_FILE) else 0
bundle = load_bundle(MODEL_VER)
MODEL_VER = os.path.getmtime(MODEL_FILE) if os.path.exists(MODEL_FILE) else 1

PAT_COLS = ["patient_id", "name", "age", "sex", "side"]
SES_COLS = (["patient_id", "session_no", "date", "notes"] + FEATURES +
            ["p_arth", "p_oa", "p_ra", "risk", "risk_score", "confirmed"])


def seed_demo():
    """4 virtual demo patients with 5 past sessions each (synthetic)."""
    rng = np.random.default_rng(11)
    demo = [("P-1001", "Ravi Kumar", "M", "Right", 1, 0.20, 0.70),   # OA, worsening
            ("P-1002", "Anita Rao", "F", "Left", 2, 0.20, 0.65),     # RA, progressing
            ("P-1003", "Meera Shetty", "F", "Right", 0, 0.0, 0.0),   # healthy
            ("P-1004", "Kiran Gowda", "F", "Both", 2, 0.85, 0.30)]   # RA, improving on treatment
    today = datetime.date.today()
    pats, rows, n = [], [], 5
    for pid, name, sex, side, cond, s0, s1 in demo:
        base = core.generate(1, rng, cond, s0 if cond else None).iloc[0]
        demo_keys = ["age", "sex_female", "bmi", "prior_injury", "family_autoimmune", "smoker"]
        pats.append(dict(patient_id=pid, name=name, age=int(base["age"]), sex=sex, side=side))
        for i in range(n):
            sev = s0 + (s1 - s0) * i / (n - 1)
            r = core.generate(1, rng, cond, sev if cond else None).iloc[0]
            vals = {k: core.cast(k, base[k] if k in demo_keys else r[k]) for k in FEATURES}
            a = core.assess(bundle, vals)
            row = dict(patient_id=pid, session_no=i + 1,
                       date=str(today - datetime.timedelta(days=21 * (n - 1 - i))),
                       notes="Demo history (synthetic)")
            row.update(vals)
            row.update(p_arth=round(a["p_arth"], 4), p_oa=round(a["p_oa"], 4), p_ra=round(a["p_ra"], 4),
                       risk=a["risk"], risk_score=round(a["risk_score"], 1), confirmed="")
            rows.append(row)
    pd.DataFrame(pats, columns=PAT_COLS).to_csv(PAT_FILE, index=False)
    pd.DataFrame(rows, columns=SES_COLS).to_csv(SES_FILE, index=False)


if not (os.path.exists(PAT_FILE) and os.path.exists(SES_FILE)):
    seed_demo()


def load_real():
    if os.path.exists(REAL_FILE):
        return pd.read_csv(REAL_FILE)
    return pd.DataFrame(columns=FEATURES + ["condition", "risk", "patient_id", "date"])


# ================================================================== state + callbacks
for k in FEATURES:
    st.session_state[f"in_{k}"] = coerce(k, st.session_state.get(f"in_{k}", core.default_values()[k]))


def apply_vals(vals, msg=None):
    for k, v in vals.items():
        if k in core.SPEC:
            st.session_state[f"in_{k}"] = core.cast(k, v)
    if msg:
        st.session_state["flash"] = msg


def make_virtual():
    cond = st.session_state.get("vp_cond", 1)
    sev = st.session_state.get("vp_sev", 0.5)
    df = core.generate(1, np.random.default_rng(), cond, sev)
    apply_vals({k: df.iloc[0][k] for k in FEATURES}, "Virtual patient generated (simulated inputs).")


patients = pd.read_csv(PAT_FILE)
sessions = pd.read_csv(SES_FILE)
ids = patients["patient_id"].tolist()
if "pending_pid" in st.session_state:                       # set by QR scan / add-patient
    st.session_state["selected_pid"] = st.session_state.pop("pending_pid")
if st.session_state.get("selected_pid") not in ids:
    st.session_state.pop("selected_pid", None)

# ================================================================== sidebar
pid = None
if ids:
    names = dict(zip(patients["patient_id"], patients["name"]))
    pid = st.sidebar.selectbox(tr("select_patient"), ids, key="selected_pid",
                               format_func=lambda x: f"{x}  ·  {names[x]}")

with st.sidebar.expander("🧪 " + tr("vp_title")):
    st.selectbox("Condition", [0, 1, 2], key="vp_cond", index=1,
                 format_func=lambda c: tr(["cond_none", "cond_oa", "cond_ra"][c]))
    st.slider(tr("vp_sev"), 0.1, 0.95, value=0.5, key="vp_sev")
    st.caption("Healthy: level of everyday aches. OA / RA: disease severity.")
    st.button(tr("vp_btn"), on_click=make_virtual, key="vp_go")
    st.caption("Fills all 3 modalities with simulated realistic values - a demo of the "
               "device inputs, not a real patient.")

if st.sidebar.checkbox("📷 " + tr("qr_scan"), key="qr_open"):
    if not CV_OK:
        st.sidebar.error("pip install opencv-python scipy")
    else:
        shot = st.sidebar.camera_input("QR", key="qr_cam")
        up = st.sidebar.file_uploader("QR image", type=["png", "jpg", "jpeg"], key="qr_up")
        src = shot or up
        if src is not None:
            data = src.getvalue()
            digest = hashlib.md5(data).hexdigest()
            img = kv.bytes_to_bgr(data)
            text = kv.decode_qr(img) if img is not None else None
            if text is None:
                st.sidebar.warning("No QR code detected. Try better light / hold it closer.")
            elif text not in ids:
                st.sidebar.error(f"QR read ({text}) but no such patient.")
            else:
                st.sidebar.success(f"Patient {text} found.")
                if st.session_state.get("qr_last") != digest:
                    st.session_state["qr_last"] = digest
                    st.session_state["pending_pid"] = text
                    st.rerun()

if pid is not None and CV_OK:
    with st.sidebar.expander("🔲 QR card"):
        png = kv.make_qr_png(pid)
        st.image(png, width=160, caption=pid)
        st.download_button("PNG", png, file_name=f"{pid}_qr.png", mime="image/png")

with st.sidebar.expander("➕ " + tr("add_patient")):
    with st.form("new_patient", clear_on_submit=True):
        n_name = st.text_input(tr("full_name"))
        n_sex = st.selectbox("M / F", ["F", "M"])
        n_side = st.selectbox(tr("side"), ["Right", "Left", "Both"])
        if st.form_submit_button(tr("add_btn")):
            if n_name.strip():
                nums = [int(x.split("-")[1]) for x in patients["patient_id"]
                        if str(x).startswith("P-") and str(x).split("-")[1].isdigit()]
                new_id = f"P-{(max(nums) + 1) if nums else 1001}"
                patients = pd.concat([patients, pd.DataFrame([dict(
                    patient_id=new_id, name=n_name.strip(), age=45, sex=n_sex, side=n_side)])],
                    ignore_index=True)
                patients.to_csv(PAT_FILE, index=False)
                st.session_state["pending_pid"] = new_id
                st.session_state["flash"] = f"{new_id} ✔"
                st.rerun()

if st.sidebar.button(tr("reset_demo")):
    seed_demo()
    st.session_state["flash"] = "Demo data restored."
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ " + tr("disclaimer"))
if DEPLOYED:
    st.sidebar.info("🌐 Public demo: synthetic data only. Do not enter real patient information. "
                    "Saved sessions are shared by all visitors and reset when the server restarts. "
                    "Live webcam, retraining and dataset upload work only when run on your own laptop.")

# ================================================================== header
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))
if pid is None:
    st.info(tr("add_patient"))
    st.stop()

prow = patients[patients["patient_id"] == pid].iloc[0]
my = sessions[sessions["patient_id"] == pid].sort_values("session_no")

if st.session_state.get("last_pid") != pid:                   # new patient -> basic demographics
    st.session_state["in_age"] = core.cast("age", prow["age"])
    st.session_state["in_sex_female"] = 1 if prow["sex"] == "F" else 0
    st.session_state["last_pid"] = pid

last_date = my["date"].iloc[-1] if len(my) else "-"
st.markdown(
    f"""<div class="hero"><h2>{html.escape(str(prow['name']))}</h2>
    <div class="sub">{tr('app_title')} · {tr('tagline')}</div>
    <span class="pill">{pid}</span><span class="pill">{prow['sex']} · {prow['age']}</span>
    <span class="pill">{tr('side')}: {prow['side']}</span>
    <span class="pill">{tr('sessions')}: {len(my)}</span>
    <span class="pill">{tr('last_assess')}: {last_date}</span></div>""", unsafe_allow_html=True)
strip = st.empty()

t_q, t_w, t_g, t_r, t_h, t_d, t_a = st.tabs([
    "🧾 " + tr("tab_quest"), "⌚ " + tr("tab_wear"), "🚶 " + tr("tab_gait"),
    "🧠 " + tr("tab_result"), "📈 " + tr("tab_history"), "🗄️ " + tr("tab_data"), "ℹ️ " + tr("tab_about")])


# ================================================================== input widgets
def widget(k, container=st):
    sp, key, lab = core.SPEC[k], f"in_{k}", label(k)
    labs_on = bool(coerce("labs_known", st.session_state.get("in_labs_known", 0)))
    dis = k in ("crp_mgL", "esr_mmhr", "rf_pos", "ccp_pos") and not labs_on
    kind = sp["kind"]
    if k == "age":
        return container.number_input(lab, 18, 90, step=1, key=key)
    if k == "bmi":
        return container.number_input(lab, 15.0, 50.0, step=0.5, key=key)
    if kind == "likert":
        out = container.slider(lab, 0, 4, step=1, key=key)
        container.caption("▸ " + tr(f"{sp['scale']}_{coerce(k, out)}"))
        return out
    if kind == "binary":
        if k == "sex_female":
            out = container.toggle("♀ " + tr("female"), key=key)
            container.caption("▸ " + (tr("female") if coerce(k, out) else tr("male")))
            return out
        return container.toggle(lab, key=key, disabled=dis)
    return container.slider(lab, sp["lo"], sp["hi"], step=sp["step"], key=key, disabled=dis)


def section(title, keys):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        cols = st.columns(2)
        for i, k in enumerate(keys):
            widget(k, cols[i % 2])


GROUPS = {g: [f for f in core.MODALITIES["quest"] if core.SPEC[f]["group"] == g]
          for g in ("history", "pain", "function", "stiff", "joints", "labs")}

with t_q:
    section("1 · " + tr("sec_history"), GROUPS["history"])
    section("2 · " + tr("sec_pain"), GROUPS["pain"])
    section("3 · " + tr("sec_function"), GROUPS["function"])
    section("4 · " + tr("sec_stiff"), GROUPS["stiff"])
    section("5 · " + tr("sec_joints"), GROUPS["joints"])
    section("6 · " + tr("sec_labs"), GROUPS["labs"])

with t_w:
    st.caption("ESP32 wearable: flex sensor + 2x MPU6050 (thigh/shank) + skin temperature + strain band. "
               "Type values, import a CSV from the device logger, or use the virtual patient.")
    section("⌚ " + tr("tab_wear"), core.MODALITIES["wear"])
    with st.expander("📂 CSV import (device logger)"):
        wcsv = st.file_uploader("CSV", type=["csv"], key="wear_csv")
        tmpl = pd.DataFrame([{k: core.default_values()[k] for k in core.MODALITIES["wear"]}])
        st.download_button("Template CSV", tmpl.to_csv(index=False).encode(), "wearable_template.csv",
                           "text/csv")
        if wcsv is not None:
            try:
                d = pd.read_csv(wcsv)
                cols = [c for c in core.MODALITIES["wear"] if c in d.columns]
                if cols:
                    vals = {c: float(pd.to_numeric(d[c], errors="coerce").median()) for c in cols}
                    st.dataframe(pd.DataFrame([vals]))
                    st.button("Use these readings", on_click=apply_vals,
                              args=(vals, f"Wearable CSV applied ({len(d)} rows, median used)."),
                              key="wear_apply")
                else:
                    st.warning("No wearable columns found. Use the template.")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

with t_g:
    section("🚶 " + tr("tab_gait"), core.MODALITIES["gait"])
    if not CV_OK:
        st.error("Camera analysis needs OpenCV + SciPy:  pip install opencv-python scipy")
    else:
        st.markdown("**Stickers:** 🟢 hip · 🟡 knee · 🔵 ankle (side view, good light, plain background)")
        sub_p, sub_l, sub_v = st.tabs([tr("cam_photo_tab"), tr("cam_live_tab"), tr("cam_video_tab")])

        with sub_p:
            shot = st.camera_input(tr("cam_open"), key="cam_photo") \
                if st.checkbox("📷 " + tr("cam_open"), key="cam_open_chk") else None
            up = st.file_uploader("Photo", type=["png", "jpg", "jpeg"], key="cam_img_up")
            src = shot or up
            if src is not None:
                img = kv.bytes_to_bgr(src.getvalue())
                pts = kv.detect_markers(img) if img is not None else None
                if pts is None:
                    st.warning("Could not find all 3 coloured markers (green, yellow, blue).")
                else:
                    ang = kv.knee_flexion_deg(pts)
                    st.image(kv.annotate(img, pts, ang)[:, :, ::-1], width=460)
                    st.metric(tr("cam_angle"), f"{ang:.0f}°")
                    st.button(tr("cam_apply_rom"), on_click=apply_vals,
                              args=({"flexion_rom_deg": ang}, "Flexion ROM updated from photo."),
                              key="apply_rom")

        with sub_l:
            st.caption("Stand side-on ~2-3 m from the camera and walk back and forth. "
                       "Reads the webcam of the computer running this app.")
            cam_no = st.number_input("Camera #", min_value=0, max_value=5, value=0, step=1, key="live_cam_no")
            dur = st.slider("Seconds", 5, 60, value=20, key="live_dur")
            b1, b2 = st.columns(2)
            go = b1.button("▶ " + tr("live_start"), type="primary", key="live_go", disabled=DEPLOYED)
            b2.button("⏹ " + tr("live_stop"), key="live_halt")
            frame_ph, metrics_ph, chart_ph = st.empty(), st.empty(), st.empty()
            if go:
                st.session_state["live_angles"] = None
                try:
                    gen = kv.live_stream(int(cam_no), max_seconds=float(dur))
                    try:
                        for n_f, ev in enumerate(gen):
                            st.session_state["live_angles"] = list(ev["angles"])
                            st.session_state["live_fps"] = max(ev["fps"], 1.0)
                            st.session_state["live_best"] = ev["best_frame"]
                            frame_ph.image(ev["frame"][:, :, ::-1], width=460)
                            if n_f % 6 == 0:
                                part = kv.analyze_series(np.array(ev["angles"]), max(ev["fps"], 1.0))
                                with metrics_ph.container():
                                    q1, q2, q3 = st.columns(3)
                                    q1.metric(tr("cam_angle"), "-" if np.isnan(ev["angle"]) else f"{ev['angle']:.0f}°")
                                    q2.metric("Strides", part.get("n_peaks", 0) if part["ok"] else 0)
                                    q3.metric("Peak", f"{part['peak_flexion']:.0f}°" if part["ok"] else "-")
                                chart_ph.line_chart(pd.DataFrame(
                                    {"knee flexion (deg)": np.nan_to_num(ev["angles"][-150:])}))
                    finally:
                        gen.close()
                except ValueError:
                    st.error(tr("live_nocam"))
            la = st.session_state.get("live_angles")
            if la and len(la) > 10:
                lres = kv.analyze_series(np.array(la), st.session_state.get("live_fps", 15.0),
                                         st.session_state.get("live_best"))
                if not lres["ok"]:
                    st.warning(f"Markers detected in too few frames ({lres['detected_pct']:.0f}%).")
                else:
                    r1c, r2c, r3c, r4c = st.columns(4)
                    r1c.metric(label("gait_knee_peak_deg"), f"{lres['peak_flexion']:.0f}°")
                    r2c.metric(label("stride_cv_pct"), "-" if lres["stride_cv"] is None else f"{lres['stride_cv']:.1f}%")
                    r3c.metric(label("cadence_spm"), "-" if lres["cadence"] is None else f"{lres['cadence']:.0f}")
                    r4c.metric("Strides", lres["n_peaks"])
                    send = {"gait_knee_peak_deg": lres["peak_flexion"]}
                    if lres["stride_cv"] is not None:
                        send["stride_cv_pct"] = lres["stride_cv"]
                    if lres["cadence"] is not None:
                        send["cadence_spm"] = lres["cadence"]
                    st.button(tr("cam_apply_gait"), on_click=apply_vals,
                              args=(send, "Gait values sent to the model inputs."), key="apply_live")

        with sub_v:
            vid = st.file_uploader("Video (mp4/mov/avi)", type=["mp4", "mov", "avi", "mkv"], key="cam_vid")
            if vid is not None and st.button("Analyze", key="run_vid"):
                suffix = os.path.splitext(vid.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                    tf.write(vid.getvalue())
                    tmp = tf.name
                try:
                    with st.spinner("..."):
                        st.session_state["vid_res"] = kv.analyze_video(tmp)
                except Exception as e:
                    st.session_state["vid_res"] = {"ok": False, "error": str(e), "detected_pct": 0}
                finally:
                    os.remove(tmp)
            res_v = st.session_state.get("vid_res")
            if res_v:
                if not res_v.get("ok"):
                    st.warning(f"Markers found in too few frames ({res_v.get('detected_pct', 0):.0f}%). "
                               + res_v.get("error", ""))
                else:
                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric(label("gait_knee_peak_deg"), f"{res_v['peak_flexion']:.0f}°")
                    a2.metric(label("stride_cv_pct"), "-" if res_v["stride_cv"] is None else f"{res_v['stride_cv']:.1f}%")
                    a3.metric(label("cadence_spm"), "-" if res_v["cadence"] is None else f"{res_v['cadence']:.0f}")
                    a4.metric("Strides", res_v["n_peaks"])
                    st.line_chart(pd.DataFrame({"knee flexion (deg)": res_v["series"]}))
                    send = {"gait_knee_peak_deg": res_v["peak_flexion"]}
                    if res_v["stride_cv"] is not None:
                        send["stride_cv_pct"] = res_v["stride_cv"]
                    if res_v["cadence"] is not None:
                        send["cadence_spm"] = res_v["cadence"]
                    st.button(tr("cam_apply_gait"), on_click=apply_vals,
                              args=(send, "Gait values sent to the model inputs."), key="apply_vid")
        st.caption("Step asymmetry needs both legs and is entered manually above.")

# ================================================================== run the models
values = {k: coerce(k, st.session_state.get(f"in_{k}")) for k in FEATURES}


@st.cache_data(show_spinner=False)
def run_model(vt, ver, _bundle):
    vals = dict(zip(FEATURES, vt))
    return core.assess(_bundle, vals), core.explain(_bundle, vals)


res, exp = run_model(tuple(values[k] for k in FEATURES), MODEL_VER, bundle)
pos = res["p_arth"] >= 0.5
lr = res["p_ra_given"]
lean = "ra" if lr >= 0.65 else ("oa" if lr <= 0.35 else "mix")
risk = res["risk"]
rcol = COLORS[risk]


def bar(pct, color):
    return f'<div class="bar"><div style="width:{max(0, min(100, pct)):.0f}%;background:{color}"></div></div>'


# ---- always-visible fusion strip
mod_delta = exp["modality"]
with strip.container():
    c0, c1, c2, c3 = st.columns([1.35, 1, 1, 1])
    vcol = "#d64545" if pos else "#2e9e5b"
    typ = ""
    if pos:
        typ = {"oa": tr("lean_oa"), "ra": tr("lean_ra"), "mix": tr("lean_mix")}[lean]
    c0.markdown(f'<div class="verdict" style="background:{vcol}"><div class="t">{tr("stage1")}</div>'
                f'<div class="v">{tr("verdict_pos") if pos else tr("verdict_neg")}</div>'
                f'<div class="t">{res["p_arth"]*100:.0f}% · {typ}</div></div>', unsafe_allow_html=True)
    for col, m, cls, nm in ((c1, "quest", "m-q", tr("tab_quest")), (c2, "wear", "m-w", tr("tab_wear")),
                            (c3, "gait", "m-g", tr("tab_gait"))):
        d = mod_delta[m] * 100
        col.markdown(f'<div class="mod {cls}"><div class="t">{nm}</div><div class="v">{d:+.0f} pts</div>'
                     f'<div class="s">{len(core.MODALITIES[m])} inputs</div></div>', unsafe_allow_html=True)


def fmt_value(k, v):
    sp = core.SPEC[k]
    if sp["kind"] == "likert":
        return tr(f"{sp['scale']}_{int(v)}")
    if sp["kind"] == "binary":
        return (tr("female") if v else tr("male")) if k == "sex_female" else (tr("yes") if v else tr("no"))
    return f"{v:g}"


# ================================================================== questionnaire summary (in questionnaire tab)
qs = core.questionnaire_scores(values)
with t_q:
    st.markdown("**📊 " + tr("sec_pain") + " / " + tr("sec_function") + " / " + tr("sec_stiff") + "**")
    for nm, key in ((tr("sec_pain"), "pain"), (tr("sec_function"), "function"), (tr("sec_stiff"), "stiffness")):
        st.markdown(f'<div class="row"><span>{nm}</span><span>{qs[key]:.0f}/100</span></div>' + bar(qs[key], "#6a4fc2"),
                    unsafe_allow_html=True)

# ================================================================== result tab
with t_r:
    st.markdown(f"### {tr('result_title')}")
    a, b = st.columns(2)
    with a:
        st.markdown(f"""<div class="card"><b>{tr('stage1')}</b>
        <div class="big" style="color:{vcol}">{res['p_arth']*100:.0f}%</div>
        <div class="small">{tr('p_arth')}</div>
        <div class="gauge"><div class="fill" style="width:{res['p_arth']*100:.0f}%;background:{vcol}"></div><div class="tick"></div></div>
        <div class="row"><span>{tr('cond_none')}: {res['p_none']*100:.0f}%</span><span>OA: {res['p_oa']*100:.0f}% · RA: {res['p_ra']*100:.0f}%</span></div>
        <p style="margin-top:10px"><b>{tr('verdict_pos') if pos else tr('verdict_neg')}</b></p></div>""",
                    unsafe_allow_html=True)
    with b:
        if pos:
            oa_p, ra_p = (1 - lr) * 100, lr * 100
            lt = {"oa": tr("lean_oa"), "ra": tr("lean_ra"), "mix": tr("lean_mix")}[lean]
            st.markdown(f"""<div class="card"><b>{tr('stage2')}</b>
            <div class="split"><div style="width:{max(oa_p,3):.0f}%;background:{C_OA}">OA {oa_p:.0f}%</div>
            <div style="width:{max(ra_p,3):.0f}%;background:{C_RA}">RA {ra_p:.0f}%</div></div>
            <p style="margin-top:10px"><b>{lt}</b></p></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="card"><b>{tr('stage2')}</b>
            <p class="small" style="margin-top:14px">-</p></div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="card"><b>{tr('risk_head')}</b><br>
    <span class="badge" style="background:{rcol}">{tr('risk_badge', level=tr(LVL_KEY[risk]).upper())}</span>
    <span class="big" style="margin-left:14px">{res['risk_score']:.0f}</span><span class="small"> / 100 {tr('score_lbl')}</span>
    <div class="gauge"><div class="fill" style="width:{res['risk_score']:.0f}%;background:{rcol}"></div></div>
    <p style="margin-top:10px">{tr(f'adv_{risk}')}</p></div>""", unsafe_allow_html=True)
    if pos:
        st.caption("OA-leaning: weight management, strengthening / physiotherapy, orthopaedic review if function is limited. "
                   "RA-leaning: early rheumatology referral; confirm with RF, anti-CCP, CRP/ESR and joint examination.")

    st.markdown(f"#### {tr('contrib_head')}")
    mx = max(max(abs(v) for v in mod_delta.values()) * 100, 10)
    for m, cls_col, nm in (("quest", "#6a4fc2", tr("tab_quest")), ("wear", "#0f7b8a", tr("tab_wear")),
                           ("gait", "#d98b0a", tr("tab_gait"))):
        d = mod_delta[m] * 100
        st.markdown(f'<div class="row"><span>{nm}</span><span>{d:+.0f} pts</span></div>' + bar(abs(d) / mx * 100, cls_col),
                    unsafe_allow_html=True)
    st.caption("Occlusion test: how much the arthritis likelihood drops when that modality is replaced by "
               "healthy reference values. Contributions overlap, so they need not add up to the total.")

    feat = exp["features"]
    st.markdown(f"#### {tr('drivers_head')}")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"**{tr('p_arth')} ↑**")
        top = feat.sort_values("d_arth", ascending=False).head(6)
        shown = False
        for _, r in top.iterrows():
            if r["d_arth"] > 0.002:
                shown = True
                st.markdown(f'<div class="drv">▲ {label(r["feature"])} = <b>{fmt_value(r["feature"], values[r["feature"]])}</b>'
                            f' <span class="small">({r["d_arth"]*100:+.1f} pts)</span></div>', unsafe_allow_html=True)
        if not shown:
            st.markdown("-")
    with d2:
        st.markdown(f"**OA ⟷ RA**" if pos else "**OA ⟷ RA**")
        if pos:
            ra_up = feat.sort_values("d_ra", ascending=False).head(3)
            oa_up = feat.sort_values("d_ra", ascending=True).head(3)
            for _, r in ra_up.iterrows():
                if r["d_ra"] > 0.002:
                    st.markdown(f'<div class="drv" style="border-left:4px solid {C_RA};padding-left:8px">RA ← {label(r["feature"])} = '
                                f'<b>{fmt_value(r["feature"], values[r["feature"]])}</b></div>', unsafe_allow_html=True)
            for _, r in oa_up.iterrows():
                if r["d_ra"] < -0.002:
                    st.markdown(f'<div class="drv" style="border-left:4px solid {C_OA};padding-left:8px">OA ← {label(r["feature"])} = '
                                f'<b>{fmt_value(r["feature"], values[r["feature"]])}</b></div>', unsafe_allow_html=True)
        else:
            st.markdown("-")

    st.markdown(f"#### {tr('checklist_head')}")
    oa_l, ra_l = core.checklist(values)
    oa_n, ra_n = sum(1 for _, m in oa_l if m), sum(1 for _, m in ra_l if m)
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(f"**OA features: {oa_n}/{len(oa_l)}**")
        for t, m in oa_l:
            st.markdown(f'<div class="chk">{"✅" if m else "▫️"} {t}</div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f"**RA features: {ra_n}/{len(ra_l)}**")
        for t, m in ra_l:
            st.markdown(f'<div class="chk">{"✅" if m else "▫️"} {t}</div>', unsafe_allow_html=True)
    rule = "RA" if ra_n / len(ra_l) > oa_n / len(oa_l) + 0.10 else ("OA" if oa_n / len(oa_l) > ra_n / len(ra_l) + 0.10 else "mixed")
    ml_l = {"ra": "RA", "oa": "OA", "mix": "mixed"}[lean] if pos else "-"
    if pos:
        st.info(f"Rule-based cross-check leans **{rule}** · model leans **{ml_l}** → "
                + ("✅ consistent" if rule == ml_l else "⚠️ differ - clinical review advised"))
    st.caption("Simplified checklist inspired by ACR knee-OA and ACR/EULAR RA features; not a classification tool.")

    # ---- save
    st.markdown(f"#### {tr('save_head')}")
    notes = st.text_input(tr("notes"), key="notes_in")
    with st.expander("🔬 Add to training dataset (clinician-confirmed cases only)"):
        st.caption("Only real patients with a clinician-confirmed diagnosis can improve the model. "
                   "Virtual patients and unconfirmed readings must NOT be added.")
        real_ok = st.checkbox("This is a real patient reading", key="real_ok", disabled=DEPLOYED)
        conf = st.selectbox("Confirmed diagnosis", [-1, 0, 1, 2], key="conf_dx",
                            format_func=lambda c: "—" if c < 0 else tr(["cond_none", "cond_oa", "cond_ra"][c]))
        conf_risk = st.selectbox("Confirmed risk level", [0, 1, 2], index=1, key="conf_risk",
                                 format_func=lambda c: tr(LVL_KEY[c]))
    if st.button("💾 " + tr("save_btn"), type="primary", key="save_btn"):
        next_no = int(my["session_no"].max()) + 1 if len(my) else 1
        row = dict(patient_id=pid, session_no=next_no, date=str(datetime.date.today()), notes=notes)
        row.update(values)
        row.update(p_arth=round(res["p_arth"], 4), p_oa=round(res["p_oa"], 4), p_ra=round(res["p_ra"], 4),
                   risk=risk, risk_score=round(res["risk_score"], 1),
                   confirmed="" if conf < 0 else ["none", "OA", "RA"][conf])
        pd.concat([sessions, pd.DataFrame([row], columns=SES_COLS)], ignore_index=True).to_csv(SES_FILE, index=False)
        msg = tr("saved", n=next_no, name=prow["name"])
        if real_ok and conf >= 0:
            rr = dict(values)
            rr.update(condition=conf, risk=conf_risk, patient_id=pid, date=str(datetime.date.today()))
            pd.concat([load_real(), pd.DataFrame([rr])], ignore_index=True).to_csv(REAL_FILE, index=False)
            msg += "  +1 confirmed case added to the training dataset."
        st.session_state["flash"] = msg
        st.rerun()

# ================================================================== history tab
with t_h:
    if len(my) == 0:
        st.info(tr("no_sessions"))
    else:
        first, latest = float(my["risk_score"].iloc[0]), float(my["risk_score"].iloc[-1])
        ch = latest - first
        trend = tr("tr_none") if len(my) < 2 else (tr("tr_worse") if ch > 10 else tr("tr_better") if ch < -10 else tr("tr_stable"))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(tr("sessions"), len(my))
        m2.metric(tr("p_arth"), f"{my['p_arth'].iloc[-1]*100:.0f}%", delta=f"{(my['p_arth'].iloc[-1]-my['p_arth'].iloc[0])*100:+.0f}", delta_color="inverse")
        m3.metric(tr("score_lbl"), f"{latest:.0f}", delta=f"{ch:+.0f}", delta_color="inverse")
        m4.metric(tr("trend"), trend)
        st.markdown(f"**{tr('risk_over')}**")
        h = my.set_index("session_no")
        st.line_chart(pd.DataFrame({tr("p_arth") + " %": h["p_arth"] * 100, tr("score_lbl"): h["risk_score"],
                                    "RA %": h["p_ra"] / (h["p_oa"] + h["p_ra"] + 1e-9) * 100}))
        show = my[["session_no", "date", "p_arth", "risk", "risk_score", "confirmed", "notes"]].copy()
        show["risk"] = show["risk"].map(lambda r: tr(LVL_KEY[int(r)]))
        show["p_arth"] = (show["p_arth"] * 100).round(0)
        st.dataframe(show)
        st.download_button("⬇️ CSV", my.to_csv(index=False).encode("utf-8"), file_name=f"{pid}_sessions.csv", mime="text/csv")

# ================================================================== data & model tab
with t_d:
    mt = bundle["metrics"]
    meta = bundle["meta"]
    st.markdown("### 🧠 Model performance (synthetic hold-out)")
    st.warning("Measured on SIMULATED patients held out from training. It shows that the fusion pipeline works - "
               "it is NOT clinical accuracy.")
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Screening AUC", f"{mt['screen_auc']:.2f}")
    g2.metric("Sensitivity", f"{mt['sensitivity']*100:.0f}%")
    g3.metric("Specificity", f"{mt['specificity']*100:.0f}%")
    g4.metric("OA vs RA acc.", f"{mt['type_acc']*100:.0f}%")
    g5.metric("Risk level acc.", f"{mt['risk_acc']*100:.0f}%")
    if "ablation" in mt:
        st.markdown("**Why fusion? Same test set, different inputs**")
        ab = pd.DataFrame(mt["ablation"]).T.rename(columns={"screen_auc": "Screening AUC", "cond_acc": "3-way accuracy"})
        st.bar_chart(ab)
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown("**Share of model importance by modality**")
        st.bar_chart(pd.Series(mt["importance_modality"]).rename(index={"quest": "Questionnaire", "wear": "Wearable", "gait": "Gait camera"}))
    with c_b:
        st.markdown("**Confusion matrix (rows = true, cols = predicted)**")
        cm = pd.DataFrame(mt["cond_confusion"], index=["True: none", "True: OA", "True: RA"], columns=["none", "OA", "RA"])
        st.dataframe(cm)

    st.markdown("### 🗄️ Dataset")
    real = load_real()
    n_real = len(real)
    d1, d2, d3 = st.columns(3)
    d1.metric("Synthetic patients", f"{meta['n_synthetic']:,}")
    d2.metric("Confirmed real cases", n_real)
    d3.metric("Model trained", meta["trained_at"][:16].replace("T", " "))
    st.caption("The synthetic generator encodes literature-style patterns (OA: older, higher BMI, short morning stiffness, "
               "crepitus, asymmetric; RA: younger, female, long morning stiffness, symmetric small-joint involvement, "
               "warmth, raised CRP/ESR, RF/anti-CCP) with overlap, atypical cases and noise.")
    up_l = st.file_uploader("Add labelled recordings (CSV with the 42 feature columns + `condition` 0/1/2 [+ `risk` 0/1/2])",
                            type=["csv"], key="label_csv", disabled=DEPLOYED)
    if up_l is not None and st.button("Add to real dataset", key="add_real"):
        try:
            d = pd.read_csv(up_l)
            miss = [c for c in FEATURES + ["condition"] if c not in d.columns]
            if miss:
                st.error(f"Missing columns: {miss[:6]}{'...' if len(miss) > 6 else ''}")
            else:
                for k in FEATURES:
                    d[k] = d[k].map(lambda v, k=k: core.cast(k, v))
                if "risk" not in d.columns:
                    d["risk"] = np.nan
                d = d[FEATURES + ["condition", "risk"]].assign(patient_id="upload", date=str(datetime.date.today()))
                pd.concat([real, d], ignore_index=True).to_csv(REAL_FILE, index=False)
                st.session_state["flash"] = f"{len(d)} labelled rows added."
                st.rerun()
        except Exception as e:
            st.error(f"Could not add: {e}")
    st.download_button("Template CSV (labelled)", pd.DataFrame([{**core.default_values(), "condition": 0, "risk": 0}]).to_csv(index=False).encode(),
                       "labelled_template.csv", "text/csv")
    use_real = n_real >= 10
    st.info(f"Retraining uses the synthetic set" + (f" + {n_real} confirmed real cases (weighted x8)." if use_real
            else f". At least 10 confirmed real cases are needed before real data is included (you have {n_real})."))
    if st.button("🔄 Retrain model now", key="retrain", disabled=DEPLOYED):
        with st.spinner("Training (about 1 minute) ..."):
            syn = pd.read_csv(SYN_FILE) if os.path.exists(SYN_FILE) else core.generate(12000, np.random.default_rng(42))
            rl = real.dropna(subset=["condition"]).copy() if use_real else None
            nb = core.train_all(syn, rl, ablation=False)
            if "ablation" in mt:
                nb["metrics"]["ablation"] = mt["ablation"]
            joblib.dump(nb, MODEL_FILE, compress=3)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state["flash"] = "Model retrained and reloaded."
        st.rerun()

# ================================================================== about tab
with t_a:
    st.markdown("""
### ArthroKNEE - multimodal early arthritis screening

**Three data sources are fused:**
1. **Clinical questionnaire** (history, pain, function, stiffness pattern, joint distribution, optional blood tests)
2. **Wearable sensors** (ESP32: flex sensor, 2x MPU6050 IMU, skin temperature, strain band): knee ROM, asymmetry,
   sway, warmth, swelling, sit-to-stand time, crepitus index
3. **Camera gait analysis** (OpenCV): peak knee flexion, stride variability, cadence (+ step asymmetry)

**Three models answer three questions** (Random Forests):
- **Stage 1** - is an arthritis pattern likely? (screening)
- **Stage 2** - if so, does it lean towards **OA** (degenerative) or **RA** (inflammatory)?
- **Risk level** - Low / Moderate / High, with a 0-100 score = 100 x (0.5 x P(Moderate) + P(High))

**Explainability:** occlusion analysis shows how much each modality and each input moved the result;
a simplified rule-based checklist (inspired by ACR knee-OA and ACR/EULAR RA features) gives an independent cross-check.

**Important limitations - please read**
- The model is trained on **synthetic patients** generated from literature-style assumptions. The metrics show the
  software pipeline works; they are **not** clinical performance.
- The simulation *assumes* self-report is subjective (reporting bias, everyday aches) while sensors measure objective
  pathology. The benefit of fusion shown here follows from those assumptions and must be re-tested on real data.
- The questionnaire is an **original draft** informed by validated instruments (e.g. WOMAC, KOOS, HAQ). Validated, licensed
  versions must be used and the tool validated before any clinical use.
- Translations are machine-written drafts; the wearable/gait labels, flags and this page are English-only.
- Not a medical device. Not a diagnostic tool. Results never replace a clinician.

**Path to real data:** (1) validate the sensors against a goniometer, (2) collect clinician-labelled cases with ethics
approval (add them in *Data & Model*), (3) retrain and evaluate on **held-out real patients**.
""")
