# spc_auto_interpret_demo_simple.py
# Streamlined Streamlit app (no CSV uploads): Nelson rules + auto-interpretation
# - Distinct synthetic data per Part–Machine combo (deterministic)
# - Demo Past Events table (optional), with option to exclude current combo
# - Sidebar filter: Past only / Generalized only / Hybrid (prefer past)
# - Hover tooltips can follow the same filter (toggleable)
# - Nelson/Western Electric/AIAG/Custom rule sets
# - Rule-colored points, Xbar and R charts, sigma bands, downloadable table

import io
import math
from datetime import datetime, timedelta
from hashlib import blake2b

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="SPC Auto-Interpretation (Nelson Rules) — Simple", layout="wide")

# ---------------------------
# Constants for Xbar–R charts
# ---------------------------
A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
D3 = {2: 0.000, 3: 0.000, 4: 0.000, 5: 0.000, 6: 0.000, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}

# --------------------------------------
# Utilities — deterministic combo variation
# --------------------------------------
def combo_variation(part: str, machine: str, seed_offset: int = 0):
    """Make each Part–Machine combo produce distinct but stable patterns."""
    key = f"{part}|{machine}".encode("utf-8")
    h = blake2b(key, digest_size=8).digest()
    a, b, c, d, e, f, g, h8 = [x for x in h]
    mean_offset = (a % 9 - 4) * 0.6        # -2.4 … +2.4 around 50
    sigma_scale = 0.8 + (b % 7) * 0.06     # 0.8 … 1.16
    alt_amp = 0.2 + (c % 5) * 0.1          # 0.2 … 0.6 alternation
    shift_span = 5 + (d % 6)               # 5–10 points
    trend_sign = 1 if (e % 2 == 0) else -1
    trend_strength = 0.03 + (f % 5) * 0.01 # 0.03…0.07 per step
    combo_seed = ((a << 24) + (b << 16) + (c << 8) + d + seed_offset) % 100000
    return mean_offset, sigma_scale, alt_amp, shift_span, trend_sign, trend_strength, combo_seed

# --------------------------------------
# Synthetic data generator (combo-aware)
# --------------------------------------
def generate_dummy_data(n_subgroups=60, subgroup_size=5, seed=42,
                        part="P1", machine="M1",
                        operator_pool=("O1","O2","O3"),
                        shift_pool=("Day","Night"), lot_pool=("L1","L2","L3")):
    mean_off, sig_scale, alt_amp, shift_span, trend_sign, trend_k, combo_seed = combo_variation(part, machine, seed_offset=seed)
    rng = np.random.default_rng(combo_seed)

    true_mean = 50.0 + mean_off
    true_sigma = 1.8 * sig_scale

    values, timestamps, meta = [], [], []
    start = datetime.now() - timedelta(hours=n_subgroups)

    for i in range(n_subgroups):
        mu = true_mean
        # shift block
        if i >= n_subgroups//3 and i < n_subgroups//3 + shift_span:
            mu += 1.8 + (sig_scale - 1.0) * 2.0
        # trend tail
        if i > n_subgroups * 0.65:
            mu += trend_sign * trend_k * (i - int(n_subgroups*0.65))
        # alternation
        mu += alt_amp if i % 2 == 0 else -alt_amp

        subgroup_vals = rng.normal(mu, true_sigma, size=subgroup_size)
        for j in range(subgroup_size):
            values.append(subgroup_vals[j])
            timestamps.append(start + timedelta(minutes=10*i + j))
            meta.append({
                "Part": part,
                "Machine": machine,
                "Operator": rng.choice(operator_pool),
                "Shift": rng.choice(shift_pool),
                "MaterialLot": rng.choice(lot_pool),
                "Subgroup": i+1,
            })

    df = pd.DataFrame(meta)
    df["Timestamp"] = timestamps
    df["Value"] = values
    return df

# --------------------------------------
# Xbar–R calc
# --------------------------------------
def xbar_r(df: pd.DataFrame, subgroup_col="Subgroup", value_col="Value", n=None):
    groups = df.groupby(subgroup_col)[value_col]
    xbar = groups.mean().rename("Xbar").reset_index()
    R = (groups.max() - groups.min()).rename("R").reset_index()
    xr = pd.merge(xbar, R, on=subgroup_col).sort_values(subgroup_col)

    if n is None:
        n = int(round(df.groupby(subgroup_col)[value_col].count().mean()))

    if n < 2 or n > 10:
        raise ValueError("Subgroup size n must be between 2 and 10 for provided constants.")

    xbarbar = xr["Xbar"].mean()
    Rbar = xr["R"].mean()

    # Xbar limits
    UCLx = xbarbar + A2[n] * Rbar
    LCLx = xbarbar - A2[n] * Rbar

    # R limits
    UCLR = D4[n] * Rbar
    LCLR = D3[n] * Rbar

    # sigma estimates (for 1σ/2σ bands)
    sigma_ind = Rbar / d2[n]
    sigma_xbar = sigma_ind / math.sqrt(n)

    xr["CLx"] = xbarbar
    xr["UCLx"] = UCLx
    xr["LCLx"] = LCLx

    xr["CLR"] = Rbar
    xr["UCLR"] = UCLR
    xr["LCLR"] = LCLR

    xr["U1x"] = xbarbar + 1*sigma_xbar
    xr["L1x"] = xbarbar - 1*sigma_xbar
    xr["U2x"] = xbarbar + 2*sigma_xbar
    xr["L2x"] = xbarbar - 2*sigma_xbar

    return xr, n, sigma_xbar

# --------------------------------------
# Nelson Rule detection (on Xbar series)
# --------------------------------------
def rule1_beyond_3sigma(x, UCL, LCL):
    return (x > UCL) | (x < LCL)

def rule2_run_same_side(x, CL, run_length=9):
    side = np.where(x > CL, 1, np.where(x < CL, -1, 0))
    out = np.zeros_like(side, dtype=bool)
    count = 1
    for i in range(1, len(side)):
        if side[i] != 0 and side[i] == side[i-1]:
            count += 1
        else:
            count = 1
        if side[i] != 0 and count >= run_length:
            out[i-run_length+1:i+1] = True
    return out

def rule3_trend(x, length=6):
    out = np.zeros_like(x, dtype=bool)
    inc = 1
    dec = 1
    for i in range(1, len(x)):
        if x[i] > x[i-1]:
            inc += 1; dec = 1
        elif x[i] < x[i-1]:
            dec += 1; inc = 1
        else:
            inc = dec = 1
        if inc >= length: out[i-length+1:i+1] = True
        if dec >= length: out[i-length+1:i+1] = True
    return out

def rule4_alternating(x, length=14):
    out = np.zeros_like(x, dtype=bool)
    alt = 1
    for i in range(2, len(x)):
        if (x[i] - x[i-1]) * (x[i-1] - x[i-2]) < 0:
            alt += 1
        else:
            alt = 1
        if alt >= length: out[i-length+1:i+1] = True
    return out

def rule5_two_of_three_outer_third(x, U2_scalar, L2_scalar):
    out = np.zeros_like(x, dtype=bool)
    for i in range(2, len(x)):
        window = x[i-2:i+1]
        high = (window > U2_scalar).sum()
        low = (window < L2_scalar).sum()
        if high >= 2 or low >= 2:
            out[i-2:i+1] = True
    return out

def rule6_four_of_five_outer_two_thirds(x, U1_scalar, L1_scalar):
    out = np.zeros_like(x, dtype=bool)
    for i in range(4, len(x)):
        window = x[i-4:i+1]
        high = (window > U1_scalar).sum()
        low = (window < L1_scalar).sum()
        if high >= 4 or low >= 4:
            out[i-4:i+1] = True
    return out

def rule7_fifteen_within_one_sigma(x, U1, L1, length=15):
    out = np.zeros_like(x, dtype=bool)
    count = 0
    for i in range(len(x)):
        if (x[i] <= U1) and (x[i] >= L1):
            count += 1
        else:
            count = 0
        if count >= length:
            out[i-length+1:i+1] = True
    return out

def rule8_eight_outside_one_sigma_none_inside(x, U1, L1, length=8):
    out = np.zeros_like(x, dtype=bool)
    count = 0
    for i in range(len(x)):
        if (x[i] >= U1) or (x[i] <= L1):
            count += 1
        else:
            count = 0
        if count >= length:
            out[i-length+1:i+1] = True
    return out

# --------------------------------------
# Interpretation knowledge base
# --------------------------------------
RULE_EXPLAIN = {
    1: "One point more than 3σ from the center line (special cause likely).",
    2: "Nine points in a row on the same side of the mean (process shift).",
    3: "Six points in a row steadily increasing or decreasing (trend).",
    4: "Fourteen points in a row alternating up/down (systematic oscillation).",
    5: "Two out of three points beyond 2σ on the same side (medium shift).",
    6: "Four out of five points beyond 1σ on the same side (small sustained shift).",
    7: "Fifteen points in a row within 1σ of the mean (reduced variation—overcontrol or data grouping).",
    8: "Eight points in a row outside 1σ with none inside (mixture or stratification).",
}
LIKELY_CAUSES = {
    1: ["Process disturbance or special event", "Measurement system issue (calibration drift)", "Incorrect control limits"],
    2: ["Setpoint change", "New material lot or supplier", "Operator changeover/procedure change"],
    3: ["Tool wear or fouling", "Gradual temperature/humidity drift", "Progressive misalignment"],
    4: ["Over-adjustment (tampering)", "Cyclic external factor (e.g., ambient conditions)", "Two alternating sources or operators"],
    5: ["Moderate mean shift", "Material property shift", "Setup change"],
    6: ["Small but sustained shift", "Bias introduced by new tooling or fixture", "Recipe change"],
    7: ["Overcontrol (adjusting every run)", "Data grouped/rounded", "Measurement resolution too coarse"],
    8: ["Mixture of populations (multiple machines, molds, lots)", "Stratification of data", "Alternating operators or cavities"],
}
ACTIONS = {
    1: ["Stop and investigate last affected subgroups", "Verify gauge calibration and remeasure", "Check recent alarms or downtime notes"],
    2: ["Review setpoints and recent change logs", "Verify material lot change and certificates", "Confirm operator followed standard work"],
    3: ["Inspect tooling for wear", "Check filters/nozzles for clogging", "Review environment logs (temp/RH)"],
    4: ["Cease over-adjustment; revert to standard settings", "Investigate cyclical factors (shifts, HVAC)", "Check scheduling or product alternation patterns"],
    5: ["Tighten process center via setup verification", "Audit material properties vs spec", "Short-term capability study"],
    6: ["Run a short trial at nominal settings", "Fixture alignment verification", "Operator retraining as needed"],
    7: ["Stop tampering; only adjust on evidence", "Increase measurement resolution if applicable", "Recompute limits if justified"],
    8: ["Stratify data by machine/lot/operator", "Run separate control charts for each stream", "Stabilize scheduling to reduce mixing"],
}

# --------------------------------------
# Sidebar controls (NO CSV)
# --------------------------------------
with st.sidebar:
    st.header("Synthetic Data")
    n_subgroups = st.number_input("Number of subgroups", 20, 200, 60, step=5)
    subgroup_size = st.number_input("Subgroup size (n)", 2, 10, 5, step=1)
    seed = st.number_input("Seed offset", 0, 10000, 42, step=1)

    st.header("Rule Set")
    rule_set = st.selectbox(
        "Choose rule set",
        ["Nelson (8 rules)", "Western Electric", "AIAG (simplified)", "Custom"],
        index=0
    )
    # Defaults
    active_rules = {r: True for r in range(1,9)}
    params = {"run_length": 9, "trend_length": 6, "alt_length": 14, "r7_length": 15, "r8_length": 8}

    if rule_set == "Western Electric":
        active_rules = {1: True, 2: True, 3: False, 4: False, 5: True, 6: True, 7: False, 8: False}
        params["run_length"] = 8
    elif rule_set == "AIAG (simplified)":
        active_rules = {1: True, 2: True, 3: True, 4: False, 5: False, 6: False, 7: False, 8: False}
        params["run_length"] = 7
        params["trend_length"] = 7
    elif rule_set == "Custom":
        st.caption("Pick the rules and tune thresholds")
        selected = st.multiselect("Active rules", options=[f"R{r}" for r in range(1,9)], default=[f"R{r}" for r in range(1,9)])
        active_rules = {r: (f"R{r}" in selected) for r in range(1,9)}
        params["run_length"] = st.slider("Run length for 'same side' (Rule 2)", 5, 12, 9)
        params["trend_length"] = st.slider("Trend length (Rule 3)", 5, 9, 6)
        params["alt_length"] = st.slider("Alternation length (Rule 4)", 10, 20, 14)
        params["r7_length"] = st.slider("Length for Rule 7 (within ±1σ)", 10, 20, 15)
        params["r8_length"] = st.slider("Length for Rule 8 (outside ±1σ)", 6, 12, 8)

    st.header("Past Events (Demo)")
    use_demo_past = st.checkbox("Use demo past events", value=True)
    include_current_in_demo = st.checkbox("Demo includes current Part+Machine", value=True)
    st.caption("• If ON, the demo history will include entries for the currently selected Part & Machine.\n"
               "• If OFF, there will be no past entries for the current combo, so you’ll see generalized guidance.")

    st.header("Interpretation Rows")
    filter_mode = st.radio(
        "Show:",
        options=["Past only", "Generalized only", "Hybrid (prefer past)"],
        index=2
    )

    st.header("Chart Options")
    show_sigma_bands = st.checkbox("Show 1σ/2σ bands", value=True)
    show_r_chart = st.checkbox("Show R chart", value=True)
    apply_filter_to_hover = st.checkbox("Apply filter mode to hover tooltips", value=True)

st.title("SPC Auto-Interpretation")

# --------------------------------------
# TOP controls – Part & Machine
# --------------------------------------
parts_known = ["P1", "P2", "P3", "P4"]
machines_known = ["M1", "M2", "M3", "M4"]
colA, colB = st.columns([1,1])
with colA:
    selected_part = st.selectbox("Part", options=parts_known, index=0)
with colB:
    selected_machine = st.selectbox("Machine", options=machines_known, index=0)

# Dashboard explanation for the current state of demo past events
if use_demo_past and include_current_in_demo:
    st.info("Past-event demo is ON and includes this Part–Machine combo. "
            "‘Hybrid’ and ‘Past only’ will use past entries when available; "
            "‘Generalized only’ will ignore past and show generic guidance.")
elif use_demo_past and not include_current_in_demo:
    st.warning("Past-event demo is ON but **excludes** this Part–Machine combo. "
               "You will see generalized guidance for this selection.")
else:
    st.caption("Past-event demo is OFF. All guidance will be generalized.")

# --------------------------------------
# Build chart data for selected combo
# --------------------------------------
df = generate_dummy_data(
    n_subgroups=int(n_subgroups),
    subgroup_size=int(subgroup_size),
    seed=int(seed),
    part=selected_part,
    machine=selected_machine
)

# --------------------------------------
# Past Events — demo only (optional)
# --------------------------------------
past_df = None
if use_demo_past:
    rows = []
    base_dates = pd.date_range("2024-01-01", periods=80, freq="7D")
    demo_map = {
        1: [("Power surge", "Check breaker; re-qualify run"),
            ("Gauge drift", "Recalibrate; remeasure"),
            ("Foreign object in feed", "Purge hopper; inspect screens")],
        2: [("Setpoint raised", "Restore nominal; lock recipe"),
            ("Offset changed", "Retrain; audit SOP"),
            ("Material grade change", "Verify COA; align parameters")],
        3: [("Tool wear", "Replace insert; adjust offsets"),
            ("Filter fouling", "Clean/replace filter"),
            ("Oven temp drift", "Re-tune controller; verify sensors")],
        4: [("Over-adjustment", "Stop tampering; follow control plan"),
            ("HVAC cycling", "Stabilize HVAC; insulate line"),
            ("Two-cavity alternation", "Separate charts per cavity")],
        5: [("Resin viscosity high", "Switch lot; adjust temp"),
            ("Setup bias after changeover", "Centerline setup; verify"),
            ("Supplier blend variation", "Quarantine lot; notify supplier")],
        6: [("Fixture bias", "Realign/qualify fixture"),
            ("Recipe tweak adds bias", "Rollback tweak; A/B confirm"),
            ("Nozzle wear bias", "Replace nozzle; check flow")],
        7: [("Overcontrol by operator", "Only adjust on evidence"),
            ("Data rounding", "Increase resolution"),
            ("Auto-tuner aggressive", "Widen deadband; retune")],
        8: [("Mixing machines", "Stratify data by machine"),
            ("Lots mixed", "Segregate lots; re-run charts"),
            ("Operator alternating", "Schedule grouping; split charts")],
    }
    idx = 0
    for r in range(1,9):
        for cause, action in demo_map[r]:
            for p in parts_known:
                for m in machines_known:
                    if (not include_current_in_demo) and (p == selected_part) and (m == selected_machine):
                        continue
                    rows.append({
                        "Date": base_dates[min(idx, len(base_dates)-1)],
                        "Part": p,
                        "Machine": m,
                        "Rule": r,
                        "Cause": f"{cause} ({p}-{m})",
                        "Action": f"{action} ({p}-{m})",
                        "Notes": f"Demo row {idx+1}"
                    })
                    idx += 1
    past_df = pd.DataFrame(rows)
    past_df["Date"] = pd.to_datetime(past_df["Date"])

# --------------------------------------
# Compute Xbar–R and Nelson rule signals
# --------------------------------------
xr, n, sigma_xbar = xbar_r(df, subgroup_col="Subgroup", value_col="Value")

X = xr["Xbar"].values
# scalar bands/CL
CL0 = float(xr["CLx"].iloc[0])
UCL0 = float(xr["UCLx"].iloc[0])
LCL0 = float(xr["LCLx"].iloc[0])
U1_0 = float(xr["U1x"].iloc[0])
L1_0 = float(xr["L1x"].iloc[0])
U2_0 = float(xr["U2x"].iloc[0])
L2_0 = float(xr["L2x"].iloc[0])

viol = {}
viol[1] = rule1_beyond_3sigma(X, UCL0, LCL0) if active_rules.get(1, False) else np.zeros_like(X, dtype=bool)
viol[2] = rule2_run_same_side(X, CL0, run_length=params["run_length"]) if active_rules.get(2, False) else np.zeros_like(X, dtype=bool)
viol[3] = rule3_trend(X, length=params["trend_length"]) if active_rules.get(3, False) else np.zeros_like(X, dtype=bool)
viol[4] = rule4_alternating(X, length=params["alt_length"]) if active_rules.get(4, False) else np.zeros_like(X, dtype=bool)
viol[5] = rule5_two_of_three_outer_third(X, U2_0, L2_0) if active_rules.get(5, False) else np.zeros_like(X, dtype=bool)
viol[6] = rule6_four_of_five_outer_two_thirds(X, U1_0, L1_0) if active_rules.get(6, False) else np.zeros_like(X, dtype=bool)
viol[7] = rule7_fifteen_within_one_sigma(X, U1_0, L1_0, length=params["r7_length"]) if active_rules.get(7, False) else np.zeros_like(X, dtype=bool)
viol[8] = rule8_eight_outside_one_sigma_none_inside(X, U1_0, L1_0, length=params["r8_length"]) if active_rules.get(8, False) else np.zeros_like(X, dtype=bool)

viol_df = pd.DataFrame({"Subgroup": xr["Subgroup"].values})
for r in range(1, 9):
    viol_df[f"Rule{r}"] = viol[r]
viol_df["AnyViolation"] = viol_df[[f"Rule{r}" for r in range(1, 9)]].any(axis=1)

# --------------------------------------
# Past/generalized lookup helper (by rule)
# --------------------------------------
def get_hist_cause_action(rule_id:int, default_cause:str, default_action:str):
    """Return (cause, action, source) where source in {'past','generalized'} for current Part–Machine."""
    source = "generalized"
    cause_txt, action_txt = default_cause, default_action
    if past_df is not None and {"Part","Machine","Rule","Cause","Action"}.issubset(past_df.columns):
        filt = (
            (past_df["Part"].astype(str) == str(selected_part)) &
            (past_df["Machine"].astype(str) == str(selected_machine)) &
            (past_df["Rule"] == int(rule_id))
        )
        hits = past_df.loc[filt]
        if not hits.empty:
            hits = hits.sort_values("Date", ascending=False) if "Date" in hits.columns else hits
            cause_txt = str(hits.iloc[0].get("Cause", cause_txt))
            action_txt = str(hits.iloc[0].get("Action", action_txt))
            source = "past"
    return cause_txt, action_txt, source

# Filter-aware picker
def choose_row_by_filter(rule_id, default_cause, default_action):
    """Return (cause, action, label_source, include_row_bool) based on sidebar filter_mode."""
    if filter_mode == "Past only":
        c, a, src = get_hist_cause_action(rule_id, default_cause, default_action)
        return (c, a, src, src == "past")
    elif filter_mode == "Generalized only":
        return (default_cause, default_action, "generalized", True)
    else:  # Hybrid (prefer past)
        c, a, src = get_hist_cause_action(rule_id, default_cause, default_action)
        return (c, a, src if src == "past" else "generalized", True)

# --------------------------------------
# Charts (hover shows source tag; can follow filter)
# --------------------------------------
hover_text = []
for _, row in viol_df.iterrows():
    s = int(row['Subgroup'])
    rules_hit = [r for r in range(1,9) if row[f"Rule{r}"]]
    if rules_hit:
        details = []
        for r in rules_hit:
            if apply_filter_to_hover:
                c, a, src, _ = choose_row_by_filter(r, LIKELY_CAUSES[r][0], ACTIONS[r][0])
            else:
                c, a, src = get_hist_cause_action(r, LIKELY_CAUSES[r][0], ACTIONS[r][0])
            tag = "(from past events)" if src == "past" else "(generalized)"
            details.append(
                f"<b>R{r}</b> {tag}: {RULE_EXPLAIN[r]}<br>"
                f"<i>Cause:</i> {c}<br><i>Action:</i> {a}"
            )
        text = f"<b>Subgroup</b>: {s}<br><b>Rules</b>: " + ", ".join([f"R{r}" for r in rules_hit]) + "<br>" + "<br>".join(details)
    else:
        text = f"<b>Subgroup</b>: {s}<br>No rule violations"
    hover_text.append(text)

# Rule colors
rule_color = {1:'#d62728', 2:'#ff7f0e', 3:'#9467bd', 4:'#17becf', 5:'#e377c2', 6:'#bcbd22', 7:'#7f7f7f', 8:'#1f77b4'}

col1, col2 = st.columns([3,2])
with col1:
    st.subheader(f"X̄ Chart — Part {selected_part}, Machine {selected_machine}")
    fig = go.Figure()
    # Base line of Xbar
    fig.add_trace(go.Scatter(
        x=xr['Subgroup'], y=xr['Xbar'], mode='lines',
        hoverinfo='skip', name='X̄', line=dict(width=1)
    ))

    # Colored violation points per rule (legend shows mapping)
    for r in range(1,9):
        if not active_rules.get(r, False):
            continue
        mask = viol_df[f'Rule{r}'].values
        if mask.any():
            fig.add_trace(go.Scatter(
                x=xr['Subgroup'][mask], y=xr['Xbar'][mask], mode='markers',
                marker=dict(size=10, color=rule_color[r]), name=f'R{r}',
                text=pd.Series(hover_text)[mask], hovertemplate="%{text}<extra></extra>"
            ))
    # OK points
    ok_mask = ~viol_df['AnyViolation'].values
    if ok_mask.any():
        fig.add_trace(go.Scatter(
            x=xr['Subgroup'][ok_mask], y=xr['Xbar'][ok_mask], mode='markers',
            marker=dict(size=6, color='#bbbbbb'), name='OK',
            text=pd.Series(hover_text)[ok_mask], hovertemplate="%{text}<extra></extra>"
        ))

    # Lines
    fig.add_hline(y=float(xr['CLx'].iloc[0]), line_dash='dash', annotation_text='CL')
    fig.add_hline(y=float(xr['UCLx'].iloc[0]), line_dash='dot', annotation_text='UCL')
    fig.add_hline(y=float(xr['LCLx'].iloc[0]), line_dash='dot', annotation_text='LCL')
    if show_sigma_bands:
        fig.add_hline(y=float(xr['U1x'].iloc[0]), line_dash='dot', annotation_text='+1σ')
        fig.add_hline(y=float(xr['L1x'].iloc[0]), line_dash='dot', annotation_text='-1σ')
        fig.add_hline(y=float(xr['U2x'].iloc[0]), line_dash='dot', annotation_text='+2σ')
        fig.add_hline(y=float(xr['L2x'].iloc[0]), line_dash='dot', annotation_text='-2σ')

    fig.update_layout(
        height=460, margin=dict(l=40,r=20,t=40,b=40),
        xaxis_title='Subgroup', yaxis_title='X̄',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_r_chart:
        st.subheader("R Chart")
        figR = go.Figure()
        figR.add_trace(go.Scatter(x=xr['Subgroup'], y=xr['R'], mode='lines+markers', name='R'))
        figR.add_hline(y=float(xr['CLR'].iloc[0]), line_dash='dash', annotation_text='CL')
        figR.add_hline(y=float(xr['UCLR'].iloc[0]), line_dash='dot', annotation_text='UCL')
        figR.add_hline(y=float(xr['LCLR'].iloc[0]), line_dash='dot', annotation_text='LCL')
        figR.update_layout(height=360, margin=dict(l=40,r=20,t=30,b=40), xaxis_title='Subgroup', yaxis_title='R')
        st.plotly_chart(figR, use_container_width=True)

with col2:
    st.subheader("Detected Rule Signals (legend colors match)")
    if viol_df['AnyViolation'].any():
        long_rows = []
        for _, row in viol_df.iterrows():
            rules_hit = [r for r in range(1,9) if row[f"Rule{r}"]]
            if rules_hit:
                long_rows.append({
                    "Subgroup": int(row["Subgroup"]),
                    "Rules": ", ".join([f"R{r}" for r in rules_hit])
                })
        shown = pd.DataFrame(long_rows).sort_values("Subgroup")
        st.dataframe(shown, use_container_width=True)
    else:
        st.success("No Nelson rule violations detected.")

# --------------------------------------
# Auto-interpretation (filter respected)
# --------------------------------------
st.subheader("Automated Interpretation: Likely Root Causes & Corrective Actions")

interpret_rows = []
for _, row in viol_df.iterrows():
    rules_hit = [r for r in range(1,9) if row[f"Rule{r}"]]
    if not rules_hit:
        continue
    for r in rules_hit:
        cause_txt, action_txt, src, include = choose_row_by_filter(
            r, LIKELY_CAUSES[r][0], ACTIONS[r][0]
        )
        if not include:
            continue
        interpret_rows.append({
            "Subgroup": int(row["Subgroup"]),
            "Rule": r,
            "What it means": RULE_EXPLAIN[r],
            "Cause": cause_txt,
            "Action": action_txt,
            "Source": "past" if src == "past" else "generalized",
            "Part": selected_part,
            "Machine": selected_machine,
        })

if interpret_rows:
    interp_df = pd.DataFrame(interpret_rows).sort_values(["Subgroup", "Rule"]).reset_index(drop=True)
    st.dataframe(interp_df, use_container_width=True)

    st.subheader("Download Findings")
    csv_bytes = io.BytesIO()
    interp_df.to_csv(csv_bytes, index=False)
    st.download_button(
        label="Download interpretation as CSV",
        data=csv_bytes.getvalue(),
        file_name="spc_interpretation_simple.csv",
        mime="text/csv",
    )
else:
    st.info("Process is in-control by the selected rule set. Continue monitoring.")

# --------------------------------------
# Notes
# --------------------------------------
with st.expander("Notes & Assumptions"):
    st.markdown(
        """
- **No CSV uploads**: everything is synthetic and demo-based for quick trials.
- **Part/Machine** dropdowns drive both **chart data** and **past-event lookup**.
- Each Part–Machine combo has different **mean, sigma, alternation amplitude, shift span, and trend direction/strength** (deterministic).
- **Demo includes current Part+Machine**: when enabled (and demo past events are ON), the history contains entries for your current selection—so *Hybrid* and *Past only* will show those; *Generalized only* ignores them by design.
- Turn **Demo includes current Part+Machine** OFF to simulate “no history” for the current combo and force generalized guidance (even in Hybrid).
- You can make hover tooltips follow the same filter via the checkbox in the sidebar.
- Confirm all SPC signals before taking action.
        """
    )
