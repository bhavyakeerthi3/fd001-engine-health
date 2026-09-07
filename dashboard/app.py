"""Single-engine FD001 research dashboard: streamlit run dashboard/app.py."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import streamlit as st
from src.inference import load_artifacts, predict_prefix
from src.preprocessing import load_fd001
from src.schema import SENSOR_DESCRIPTIONS
from dashboard.charts import rul_chart, health_chart, model_chart, sensor_chart, shap_chart

st.set_page_config(page_title="AeroHealth | FD001", page_icon="✈", layout="wide", initial_sidebar_state="auto")
st.markdown(f"<style>{(ROOT/'dashboard'/'theme.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
PLOT_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


@st.cache_resource
def resources():
    return load_artifacts(ROOT)


@st.cache_data
def test_data(cap):
    return load_fd001(ROOT/"data"/"raw", cap)[1]


@st.cache_data
def recorded_history(modified_ns):
    # Exported causal predictions from the same saved models; no retrospective smoothing.
    return pd.read_csv(ROOT/"reports"/"test_predictions.csv")


@st.cache_data(max_entries=128)
def result_for(engine_id, cycle):
    bundle = resources()
    df = test_data(bundle["config"]["rul_cap"])
    prefix = df[(df.engine_id == engine_id) & (df.cycle <= cycle)].drop(columns=["rul_raw", "rul_target"])
    return predict_prefix(prefix, bundle)


def sensor_label(sensor):
    symbol, _, _ = SENSOR_DESCRIPTIONS[int(sensor.split("_")[1])-1]
    return f"{sensor.replace('sensor_', 'S')} · {symbol}"


def panel_header(kicker, title):
    st.markdown(f"<div class='panel-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.subheader(title)


if not (ROOT/"artifacts"/"pipeline.joblib").exists():
    st.title("Engine health monitor")
    st.info("Run `python -m src.pipeline` from the repository root to create the trained artifacts.")
    st.stop()
bundle = resources(); config = bundle["config"]
df = test_data(config["rul_cap"])
with st.sidebar:
    st.markdown("<div class='brand'><div class='brand-mark'>✳</div><div><div class='brand-name'>AeroHealth</div><div class='brand-sub'>ENGINE INTELLIGENCE</div></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow sidebar-eyebrow'>Observation controls</div>", unsafe_allow_html=True)
    engine = st.selectbox("Test engine", sorted(df.engine_id.unique().tolist()), format_func=lambda x:f"Engine {x:03d}")
    engine_df = df[df.engine_id == engine]
    maximum = int(engine_df.cycle.max())
    cycle = st.slider("Observed cycle", 1, maximum, maximum, key=f"cycle_{engine}")
    st.caption(f"Cycle {cycle} / {maximum} available · history ends at selection")
    st.divider()
    st.markdown("<div class='eyebrow sidebar-eyebrow'>Sensor workspace</div>", unsafe_allow_html=True)
    selected = st.multiselect("Sensors to inspect", bundle["sensors"], default=bundle["sensors"][:3], format_func=sensor_label)
    st.caption("Open Sensor trends to inspect the selected channels.")
    st.divider()
    show_truth = st.checkbox("Show evaluation-only true RUL", value=False)
    st.caption("True RUL is a held-out evaluation label. It never enters prediction.")
    st.markdown("<div class='eyebrow sidebar-eyebrow'>Experiment context</div>", unsafe_allow_html=True)
    st.caption(f"NASA C-MAPSS · FD001\n\nHistory window: {config['window']} cycles\n\nTarget ceiling: {config['rul_cap']} cycles\n\nSingle operating condition · HPC degradation")
    st.caption("Independent research portfolio. Simulated engine data.")

result = result_for(engine, cycle)
prefix = engine_df[engine_df.cycle <= cycle].copy()
history_path = ROOT/"reports"/"test_predictions.csv"
all_history = recorded_history(history_path.stat().st_mtime_ns)
history = all_history[(all_history.engine_id == engine) & (all_history.cycle <= cycle)].copy()
colors = {"Green":"#207653", "Yellow":"#896800", "Orange":"#b45113", "Red":"#b5323b"}
status = result["status"]
flagged = bool(result["anomaly_history"][-1] < 0)

st.markdown("<div class='eyebrow'>Aircraft prognostics &nbsp; / &nbsp; Single engine analysis</div>", unsafe_allow_html=True)
heading, badge = st.columns([5,1])
with heading:
    st.title("Engine health monitor")
    st.caption("From sensor history to remaining life — with the evidence behind every estimate.")
with badge:
    st.markdown("<span class='context-pill'>FD001 · RESEARCH</span>", unsafe_allow_html=True)
st.markdown(f"<div class='status-bar'><span class='engine-title'>Engine {engine:03d} <span style='color:#8b9baa;font-weight:400'> / </span> Cycle {cycle}</span><span class='status-badge' style='color:{colors[status]};background:{colors[status]}0d'><span class='status-dot'>●</span>{status} · Project status</span></div>", unsafe_allow_html=True)

cards = st.columns(4)
cards[0].metric("Health score / 100", f"{result['health_history'][-1]:.1f}", help="PCA score anchored to early and terminal training states; not a failure probability.")
cards[1].metric("Predicted RUL · cycles", f"{result['mean']:.1f}", help="Equal-weight mean of the four model predictions.")
cards[2].metric("Spread interval · cycles", f"{result['lower']:.1f}–{result['upper']:.1f}", help="Mean ± 1.96 sample standard deviations across four correlated models. Uncalibrated; no coverage guarantee.")
cards[3].metric("Anomaly reference", "Flagged" if flagged else "Within range", help="Isolation Forest compares sensor windows with an early-life training reference. This is not a diagnosed fault.")
st.caption(f"{result['window_observed']} actual measurements in current window · Relative health score · Uncalibrated model spread")
t = config["maintenance_rul_thresholds"]
bands = [("Red", f"< {t['red_below']}"), ("Orange", f"{t['red_below']}–<{t['orange_below']}"),
         ("Yellow", f"{t['orange_below']}–<{t['yellow_below']}"), ("Green", f"≥ {t['yellow_below']}")]
chips = "".join(f"<span class='policy-band {'active' if color == status else ''}' style='color:{colors[color]}'>{color} {label}</span>" for color,label in bands)
st.markdown(f"<div class='policy'><span class='policy-label'>Project policy · lower RUL bound</span>{chips}<span class='policy-label'>cycles · not aerospace limits</span></div>", unsafe_allow_html=True)

overview, sensors_tab, explanations = st.tabs(["Overview", "Sensor trends", "Explainability"])
with overview:
    primary, secondary = st.columns([1.7,1],gap="medium")
    with primary:
        with st.container(border=True, key="panel_rul"):
            panel_header("01 / Prognosis", "Remaining useful life")
            st.caption("Available history only · shaded region shows model disagreement")
            st.plotly_chart(rul_chart(history,config["rul_cap"],show_truth),width="stretch",config=PLOT_CONFIG,key="rul_history")
    with secondary:
        with st.container(border=True, key="panel_models"):
            panel_header("02 / Model agreement", "Four perspectives, one engine")
            st.caption("All models use the same available sensor window.")
            st.plotly_chart(model_chart(result,config["rul_cap"]),width="stretch",config=PLOT_CONFIG,key="model_compare")
            st.caption("Agreement can coexist with shared prediction error.")
    health, interpretation = st.columns([1.7,1],gap="medium")
    with health:
        with st.container(border=True, key="panel_health"):
            panel_header("03 / Condition", "Health trajectory")
            st.plotly_chart(health_chart(prefix,result),width="stretch",config=PLOT_CONFIG,key="health_history")
    with interpretation:
        with st.container(border=True, key="panel_interpretation"):
            panel_header("04 / Reading the evidence", "At this observation")
            st.write(f"The ensemble estimates **{result['mean']:.1f} remaining cycles**. Its lower spread bound of **{result['lower']:.1f} cycles** places this observation in the **{status}** project band.")
            st.write("The current sensor window is **unusual relative to the early-life reference**." if flagged else "The current sensor window is **within the early-life anomaly reference**.")
            st.caption("These are model outputs from simulated data, not operational maintenance instructions. Anomaly flags do not override the RUL-based status.")
    if show_truth:
        truth = float(prefix.rul_raw.iloc[-1])
        st.info(f"Evaluation-only original RUL: {truth:.1f} cycles. Capped target: {min(truth,config['rul_cap']):.1f} cycles. Neither label enters inference.")
    with st.expander("Prediction details and project assumptions"):
        st.dataframe(pd.DataFrame({"Model":list(result["predictions"]),"RUL (cycles)":list(result["predictions"].values())}),hide_index=True,width="stretch",column_config={"RUL (cycles)":st.column_config.NumberColumn(format="%.2f")})
        st.write(f"Predictions are bounded to 0–{config['rul_cap']} cycles. Short windows repeat the earliest observed measurement on the left. The heuristic spread has no guaranteed coverage. Status thresholds are illustrative project choices, not NASA or GE Aerospace limits.")
        export = pd.DataFrame([{"engine_id":engine,"cycle":cycle,"health_score":result['health_history'][-1],"ensemble_rul":result['mean'],"spread_lower":result['lower'],"spread_upper":result['upper'],"project_status":status,**result['predictions']}])
        st.download_button("Download this observation",export.to_csv(index=False),file_name=f"engine_{engine:03d}_cycle_{cycle}.csv",mime="text/csv")

with sensors_tab:
    st.subheader("Sensor behavior through time")
    st.caption("Raw measurements and a causal rolling mean. Sensor identities follow NASA's modeling paper.")
    standardized = st.toggle("Use training-standardized scale",help="Apply the saved training mean and standard deviation. No statistics are fitted to this engine.")
    if not selected:
        st.info("Select a sensor in the sidebar to inspect its observed trend.")
    columns = st.columns(2,gap="medium")
    for index,sensor in enumerate(selected):
        symbol,description,unit = SENSOR_DESCRIPTIONS[int(sensor.split("_")[1])-1]
        values = prefix[sensor]
        if standardized:
            position = bundle["sensors"].index(sensor)
            values = (values-bundle["scaler"].mean_[position])/bundle["scaler"].scale_[position]
            unit = "Training standard deviations"
        with columns[index%2]:
            with st.container(border=True, key=f"panel_sensor_{sensor}"):
                panel_header(sensor.replace("sensor_","Sensor "),f"{symbol} · {description}")
                st.plotly_chart(sensor_chart(prefix.cycle,values,values.rolling(config['window'],min_periods=1).mean(),unit),width="stretch",config=PLOT_CONFIG,key=f"trend_{sensor}")
    st.caption(f"Rolling window: {config['window']} cycles, using available past measurements only. Changing the display scale does not change model predictions.")

with explanations:
    st.subheader("What moved the XGBoost estimate?")
    st.caption("Signed SHAP contributions, aggregated from window statistics to sensors. Values explain the raw output before clipping.")
    contributions = result["shap"].rename("Contribution").rename_axis("Sensor").reset_index()
    contributions["Magnitude"] = contributions.Contribution.abs()
    explanation_chart, explanation_note = st.columns([1.7,1],gap="medium")
    with explanation_chart:
        with st.container(border=True, key="panel_shap"):
            panel_header("Sensor attribution", "Top contributors")
            st.plotly_chart(shap_chart(contributions),width="stretch",config=PLOT_CONFIG,key="shap_contributors")
    with explanation_note:
        with st.container(border=True, key="panel_directions"):
            panel_header("How to read it", "Direction matters")
            st.markdown("**Teal / positive** contributions increase the estimated remaining life.")
            st.markdown("**Orange / negative** contributions reduce the estimated remaining life.")
            st.caption("Each sensor combines its latest value, rolling mean, variability and slope. Correlated sensors can share attribution. This is an explanation of model behavior, not a causal fault diagnosis.")
    st.markdown(f"<div class='formula'>Baseline <b>{result['shap_baseline']:.2f}</b> + all sensor contributions <b>{contributions.Contribution.sum():+.2f}</b> = raw XGBoost RUL <b>{result['raw_xgb']:.2f} cycles</b></div>",unsafe_allow_html=True)
    with st.expander("All sensor contributions"):
        full = contributions.drop(columns="Magnitude").sort_values("Contribution")
        st.dataframe(full,hide_index=True,width="stretch",column_config={"Contribution":st.column_config.NumberColumn("RUL contribution (cycles)",format="%+.3f")})
        st.download_button("Download sensor contributions",full.to_csv(index=False),file_name=f"shap_engine_{engine:03d}_cycle_{cycle}.csv",mime="text/csv")

st.markdown("<div class='footnote'>AEROHEALTH &nbsp; / &nbsp; Independent research portfolio &nbsp; · &nbsp; NASA C-MAPSS FD001 &nbsp; · &nbsp; Simulated data, auditable predictions</div>",unsafe_allow_html=True)
