"""Retrofit Monitor dashboard (Streamlit).

What it does (design spec FR6 + FR7):
  - shows the machine's current state (Idle / Wash / Rinse / Drain / Spin)
  - raises a red ALERT when the fault detector flags the machine
  - shows recent history charts
  - shows total running time and idle time

There is no real machine yet, so it REPLAYS one fake wash cycle, one 1-second
window at a time, as if the data were arriving live. The two models you trained
(ml/Classifier_model.py) do the thinking:
    state model    -> "which state is the machine in?"
    fault detector -> "does this window look unusual?"

Run from the repo folder:
    python -m streamlit run dashboard/app.py
"""
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
FEATURES = ["rms_current_a", "peak_current_a", "crest_factor",
            "vib_rms_g", "vib_dom_freq_hz"]
STATE_ORDER = ["Idle", "Wash", "Rinse", "Drain", "Spin"]
STATE_COLORS = ["#9aa5b1", "#2f80ed", "#27ae9f", "#e0a526", "#8e5bd9"]

st.set_page_config(page_title="Retrofit Monitor", page_icon="🔧", layout="wide")


# ---------------- Loading (cached, so it happens once) ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "ml" / "data" / "fake_features.csv")
    df.columns = df.columns.str.strip()                 # same clean-up as the training script
    for col in ["state", "fault_type"]:
        df[col] = df[col].str.strip()
    return df


@st.cache_resource
def load_models():
    d = ROOT / "ml" / "models"
    type_file = d / "fault_type_forest.joblib"          # optional: made by Part 3
    return (joblib.load(d / "state_random_forest.joblib"),
            joblib.load(d / "fault_isolation_forest.joblib"),
            joblib.load(type_file) if type_file.exists() else None)


try:
    df = load_data()
    state_model, detector, type_model = load_models()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}\n\nRun `python ml/make_fake_dataset.py` and "
             "then `python ml/Classifier_model.py` first, to create the data and the models.")
    st.stop()


# ---------------- Sidebar controls ----------------
cycles = df.groupby("cycle_id")["fault_type"].first()
fault_ids = cycles[cycles != "none"].index.tolist()
normal_ids = cycles[cycles == "none"].index.tolist()


def cycle_label(cid):
    kind = cycles[cid]
    return f"Cycle {cid}  ({'normal' if kind == 'none' else 'FAULT: ' + kind})"


st.sidebar.header("Replay settings")
cycle_id = st.sidebar.selectbox("Wash cycle to replay", fault_ids + normal_ids,
                                format_func=cycle_label)
speed = st.sidebar.slider("Speed (windows per second)", 1, 10, 3)
need = st.sidebar.slider("Alert after N flagged windows in a row", 1, 4, 2,
                         help="1 = react instantly but more false alarms. "
                              "Higher = fewer false alarms but a slower alert.")

# ---------------- Run the models on the chosen cycle ----------------
cyc = df[df["cycle_id"] == cycle_id].reset_index(drop=True)
cyc["pred_state"] = state_model.predict(cyc[FEATURES])
cyc["flag"] = detector.predict(cyc[FEATURES]) == -1                      # -1 = unusual
cyc["alert"] = cyc["flag"].rolling(need).sum().fillna(0) >= need         # N in a row
if type_model is not None:       # chance of each fault type, one column per type
    proba = type_model.predict_proba(cyc[FEATURES])
    for i, name in enumerate(type_model.classes_):
        cyc["p_" + name] = proba[:, i]
n = len(cyc)

# ---------------- Replay state ----------------
ss = st.session_state
if ss.get("cycle") != cycle_id:          # new cycle chosen: start again
    ss.cycle, ss.t, ss.playing = cycle_id, 1, False
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶ Play"):
    if ss.t >= n:
        ss.t = 1
    ss.playing = True
if c2.button("⏸ Pause"):
    ss.playing = False
if c3.button("⏮ Reset"):
    ss.t, ss.playing = 1, False

st.title("🔧 Retrofit Monitor")
st.caption("Washing-machine model. Replaying fake sensor data through your trained models.")


def draw():
    shown = cyc.iloc[:ss.t]
    last = shown.iloc[-1]
    # Keep the alert on screen for 5 windows after the last flag, so it is not missed.
    alert_now = bool(shown["alert"].tail(5).any())

    # --- alert banner (FR6) ---
    if alert_now:
        st.error("🚨 **FAULT ALERT** — the machine is behaving abnormally. Check it now.")
        if type_model is not None:
            # average the type percentages over the flagged windows of the last 5 seconds
            recent = shown.tail(5)
            recent = recent[recent["flag"]]
            st.markdown("**Which fault is it?** (chance of each type)")
            for name in type_model.classes_:
                p = float(recent["p_" + name].mean())
                st.progress(p, text=f"{name.capitalize()}: {p:.0%}")
    else:
        st.success("✅ Machine healthy")

    # --- headline numbers (FR7) ---
    running = int((shown["pred_state"] != "Idle").sum())
    idle = int((shown["pred_state"] == "Idle").sum())
    alerts = int((shown["alert"] & ~shown["alert"].shift(fill_value=False)).sum())  # count rising edges
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current state", "FAULT" if alert_now else last["pred_state"])
    m2.metric("Running time", f"{running} s")
    m3.metric("Idle time", f"{idle} s")
    m4.metric("Fault alerts", alerts)

    # --- charts: stacked, one shared time axis, full width ---
    ticks = list(range(0, n + 1, 2))                       # a vertical line every 2 seconds
    x_dom = alt.Scale(domain=[0, n])
    LEFT = 60                                              # same left space for every chart, so they line up
    flagged = shown[shown["flag"]]
    colour = alt.Color("pred_state:N", legend=None,
                       scale=alt.Scale(domain=STATE_ORDER, range=STATE_COLORS))

    def x_axis(show_labels):
        return alt.X("window:Q", scale=x_dom,
                     axis=alt.Axis(values=ticks, grid=True, labels=show_labels,
                                   ticks=show_labels, title="Time (seconds)" if show_labels else None))

    def bands(show_labels):                                # soft state colour behind the line
        return alt.Chart(shown).mark_rect(opacity=0.22).encode(
            x=x_axis(show_labels), x2=alt.X2("window2:Q"), color=colour,
        ).transform_calculate(window2="datum.window + 1")

    def line(col, title, color, show_labels):
        y = alt.Y(f"{col}:Q", axis=alt.Axis(title=title, minExtent=LEFT))
        base = alt.Chart(shown).mark_line(color=color).encode(x=x_axis(show_labels), y=y)
        dots = alt.Chart(flagged).mark_circle(color="#e5484d", size=90).encode(x=x_axis(show_labels), y=f"{col}:Q")
        return (bands(show_labels) + base + dots).properties(height=150)

    legend = "  ".join(f"<span style='color:{c}'>■</span> {s}" for s, c in zip(STATE_ORDER, STATE_COLORS))
    st.markdown(f"**State colours:** {legend} &nbsp;|&nbsp; red dots = flagged as unusual", unsafe_allow_html=True)
    strip = alt.Chart(shown).mark_rect(height=26).encode(
        x=x_axis(False), x2=alt.X2("window2:Q"),
        y=alt.Y("label:N", axis=alt.Axis(title="State", labels=False, ticks=False, domain=False, minExtent=LEFT)),
        color=colour,
        tooltip=["window", "pred_state"],
    ).transform_calculate(window2="datum.window + 1", label="'state'").properties(height=36)

    # Three separate charts with the same left space and the same x scale, so the
    # vertical lines (every 2 s) line up from top to bottom.
    st.altair_chart(line("rms_current_a", "Current (A)", "#2f80ed", False), width="stretch")
    st.altair_chart(line("vib_rms_g", "Vibration (g)", "#8e5bd9", False), width="stretch")
    st.altair_chart(strip.encode(x=x_axis(True)).properties(height=60), width="stretch")

    with st.expander("Check the models against the true labels (only possible with fake data)"):
        t = shown[["window", "state", "pred_state", "flag", "alert"]].rename(columns={
            "state": "true state", "pred_state": "model said",
            "flag": "flagged", "alert": "alert"})
        st.dataframe(t.tail(15), hide_index=True, width="stretch")

    # --- move the replay forward ---
    if ss.playing:
        if ss.t < n:
            ss.t += 1
        else:
            ss.playing = False
            st.rerun()


interval = (1 / speed) if ss.playing else None
st.fragment(run_every=interval)(draw)()
