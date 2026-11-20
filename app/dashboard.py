"""One-page offline replay dashboard. All numbers and statuses come from the API."""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.backends import ApiBackend, BackendError

st.set_page_config(page_title="5G Network Intelligence", page_icon="◉", layout="wide")
st.markdown(
    """<style>
.stApp {background:#f5f6f8; color:#142632}
[data-testid="stSidebar"] {background:#e9eef0}
[data-testid="stSidebar"] input {color:#142632}
.block-container {padding-top:2.2rem; padding-bottom:2rem; max-width:1450px}
h1 {font-size:2.2rem !important; letter-spacing:-.06rem}
[data-testid="stMetric"] {background:white; border:1px solid #e1e6e9; border-radius:12px; padding:1rem 1.2rem}
.replay-badge {background:#e2ecea;color:#1d6658;padding:.35rem .65rem;border-radius:5px;font-size:.72rem;font-weight:700;letter-spacing:.08em}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def backend(url):
    return ApiBackend(url)


@st.cache_data(ttl=60)
def context(url):
    client = ApiBackend(url)
    return client.meta(), client.info()


url = os.getenv("API_URL", "http://127.0.0.1:8000")
client = backend(url)
st.markdown('<span class="replay-badge">HISTORICAL / OFFLINE REPLAY</span>', unsafe_allow_html=True)
st.title("5G Network Intelligence")
st.caption(
    "KPI forecasting & capacity-risk monitoring · Observe the past. Inspect the forecast. Understand the limits."
)
try:
    meta, info = context(url)
except BackendError as exc:
    st.error(str(exc))
    st.info(
        "Prepare the demo assets and start the API. No fabricated sample results are substituted."
    )
    st.stop()

with st.sidebar:
    st.header("Replay explorer")
    st.caption("A real historical subset; no live network connection.")
    if not meta["mapping_verified"]:
        st.error("BS/cell/beam mapping is unverified.")
        st.stop()
    topology = meta["topology"]
    bs = st.selectbox("Base station", sorted({parts[0] for parts in topology.values()}), key="bs")
    cell = st.selectbox(
        "Cell", sorted({parts[1] for parts in topology.values() if parts[0] == bs}), key="cell"
    )
    beams = [beam for beam, parts in topology.items() if parts[:2] == [bs, cell]]
    beam_id = st.selectbox("Beam", beams, key="beam")
    origin = st.slider(
        "Forecast origin (relative hour)",
        meta["origin_min"],
        meta["origin_max"],
        meta["origin_min"],
        key="origin",
    )
    horizon = st.selectbox(
        "Focus horizon", [1, 6, 24], format_func=lambda h: f"+{h} hours", key="horizon"
    )
    st.divider()
    st.caption(f"Loaded: {meta['beam_subset_count']} beams · 168-hour history")
    st.caption(f"Model {info['model_version']} · data {info['dataset_revision'][:12]}")

try:
    result = client.risk([beam_id], origin)
except BackendError as exc:
    st.error(str(exc))
    st.stop()
beam = result["beams"][0]
selected = next(p for p in beam["predictions"] if p["horizon"] == horizon)
st.caption(
    f"{info['champion']} · frozen version {info['model_version']} · dataset revision {info['dataset_revision'][:12]}"
)
columns = st.columns(4)
for col, label, value in zip(
    columns,
    ["Beams loaded", "Chosen model", "Focus horizon", "Historical risk status"],
    [meta["beam_subset_count"], info["champion"], f"+{horizon}h", selected["status"]],
):
    col.metric(label, value)

st.subheader(f"Beam {beam_id} · history and forecast")
st.caption(
    "168 observed hours, three independent point forecasts. Target indices are relative dataset hours."
)
figure = go.Figure()
figure.add_trace(
    go.Scatter(
        x=[p["time"] for p in beam["history"]],
        y=[p["value"] for p in beam["history"]],
        name="Observed history",
        mode="lines",
        line={"color": "#284c64", "width": 2},
        connectgaps=False,
    )
)
for prediction in beam["predictions"]:
    h = prediction["horizon"]
    figure.add_trace(
        go.Scatter(
            x=[prediction["target_index"]],
            y=[prediction["point"]],
            mode="markers",
            name=f"Forecast +{h}h",
            marker={
                "size": 14 if h == horizon else 10,
                "color": "#087e78" if h == horizon else "#97bcb8",
                "symbol": "diamond",
            },
            hovertemplate=f"+{h}h forecast: %{{y:.4f}}<extra></extra>",
        )
    )
    if prediction["actual"] is not None:
        figure.add_trace(
            go.Scatter(
                x=[prediction["target_index"]],
                y=[prediction["actual"]],
                mode="markers",
                name=f"Known actual +{h}h",
                marker={"size": 10, "color": "#64748b", "symbol": "circle-open"},
            )
        )
for field, label, color in [
    ("historical_warning_p90", "Historical Warning Threshold (P90)", "#c28c24"),
    ("historical_critical_p95", "Historical Critical Threshold (P95)", "#bd5b50"),
]:
    if beam[field] is not None:
        figure.add_hline(
            y=beam[field],
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top left" if "warning" in field else "top right",
        )
figure.add_vline(x=origin, line_dash="dot", line_color="#94a3b8")
figure.update_layout(
    height=420,
    margin={"t": 25, "b": 25, "l": 15, "r": 15},
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis_title="Relative hour",
    yaxis_title="DLPRB usage (dataset units)",
    legend={"orientation": "h", "y": -0.2},
    font={"family": "Arial", "color": "#243b4b"},
    hovermode="closest",
)
figure.update_xaxes(showgrid=False)
figure.update_yaxes(gridcolor="#edf0f2", rangemode="tozero")
st.plotly_chart(figure, width="stretch")
if not selected["valid_mask"]:
    st.warning(f"Beam/horizon excluded from risk scoring: {selected['reason']}")
if selected["actual"] is None:
    st.info("No actual is known for this target index. The point forecast remains available.")
if any(p["value"] is None for p in beam["history"]):
    st.info(
        "History contains missing observations; gaps stay missing and the champion uses native missing-value handling."
    )

left, right = st.columns([1.25, 1])
with left:
    st.subheader("Risk detail")
    st.dataframe(
        pd.DataFrame(beam["predictions"])[["horizon", "target_index", "point", "actual", "status"]],
        hide_index=True,
        width="stretch",
    )
    warning = beam["historical_warning_p90"]
    critical = beam["historical_critical_p95"]
    warning_text = f"{warning:.4f}" if warning is not None else "unavailable"
    critical_text = f"{critical:.4f}" if critical is not None else "unavailable"
    st.caption(
        f"Historical P90: {warning_text} · P95: {critical_text} · fitted through hour {beam['threshold_fit_end']}"
    )
    st.caption(
        "Thresholds are historical per-beam quantiles. They are not conditional forecast q90, congestion probabilities or verified incidents. No forecast uncertainty band is available."
    )
with right:
    st.subheader("Champion vs baseline")
    benchmark = pd.DataFrame(info["benchmark"])
    if not benchmark.empty:
        champion = benchmark[benchmark.model == info["champion"]][["horizon", "mae"]].rename(
            columns={"mae": "Champion MAE"}
        )
        baselines = benchmark[benchmark.model != info["champion"]][
            ["horizon", "model", "mae"]
        ].rename(columns={"model": "Baseline", "mae": "Baseline MAE"})
        table = champion.merge(baselines, on="horizon")
        st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Frozen full-network internal holdout MAE, not a metric computed from this 24-beam demo or selected window."
    )
    st.warning(
        "P90 recall is about 10–16%. Many exceedances are missed, despite lower point-forecast MAE."
    )
with st.expander("Provenance & limitations"):
    for limitation in info["limitations"]:
        st.write("• " + limitation)
    st.write(meta["dataset_scope"])
    st.caption(
        "NETOP GOTS-F · MIT dataset · relative hourly index; physical usage units unverified."
    )
