"""Streamlit dashboard for git-xrays — connects to the FastAPI backend."""
from __future__ import annotations

import sys

import httpx
import plotly.graph_objects as go
import streamlit as st

# ── Configuration ───────────────────────────────────────────────────

API_URL = "http://localhost:8000"
for arg in sys.argv:
    if arg.startswith("--api-url="):
        API_URL = arg.split("=", 1)[1]

st.set_page_config(page_title="git-xrays", layout="wide")


# ── Data Fetching ───────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch(endpoint: str, params: dict | None = None) -> list | dict | None:
    try:
        resp = httpx.get(f"{API_URL}{endpoint}", params=params, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(f"Cannot connect to API at {API_URL}. Is the server running?")
        st.stop()


# ── Sidebar ─────────────────────────────────────────────────────────

st.sidebar.title("git-xrays")

repos = fetch("/api/repos") or []
if not repos:
    st.sidebar.warning("No repositories found. Run `analyze-repo <path> --all` first.")
    st.stop()

selected_repo = st.sidebar.selectbox("Repository", repos)

runs = fetch("/api/runs", params={"repo": selected_repo}) or []
if not runs:
    st.sidebar.warning("No runs for this repository.")
    st.stop()


def _run_label(r: dict) -> str:
    date = str(r["created_at"])[:19]
    return f"{date}  (DX: {r['dx_score']:.4f})"


run_labels = [_run_label(r) for r in runs]
selected_idx = st.sidebar.selectbox(
    "Run", range(len(runs)), format_func=lambda i: run_labels[i]
)
selected_run_id = runs[selected_idx]["run_id"]

compare = st.sidebar.checkbox("Compare with another run")
compare_run_id = None
if compare and len(runs) > 1:
    other_runs = [(i, r) for i, r in enumerate(runs) if i != selected_idx]
    compare_idx = st.sidebar.selectbox(
        "Compare to",
        [i for i, _ in other_runs],
        format_func=lambda i: run_labels[i],
    )
    compare_run_id = runs[compare_idx]["run_id"]

# ── Load Run Detail ─────────────────────────────────────────────────

run_detail = fetch(f"/api/runs/{selected_run_id}")
if not run_detail:
    st.error("Run not found.")
    st.stop()


# ── Tab Layout ──────────────────────────────────────────────────────

tab_names = [
    "Overview", "Hotspots", "Knowledge", "Coupling",
    "Complexity", "Clustering", "Effort", "Anemia", "Time Travel",
]
tabs = st.tabs(tab_names)

# ── Tab 1: Overview ─────────────────────────────────────────────────

with tabs[0]:
    st.header("Overview")

    col1, col2 = st.columns([1, 2])

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=run_detail["dx_score"],
            title={"text": "DX Score"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 0.3], "color": "#ffcccc"},
                    {"range": [0.3, 0.6], "color": "#fff3cd"},
                    {"range": [0.6, 1.0], "color": "#d4edda"},
                ],
            },
        ))
        fig.update_layout(height=300, margin=dict(t=60, b=20, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Throughput", f"{run_detail['dx_throughput']:.4f}")
        m2.metric("Feedback", f"{run_detail['dx_feedback_delay']:.4f}")
        m3.metric("Focus", f"{run_detail['dx_focus_ratio']:.4f}")
        m4.metric("Cognitive Load", f"{run_detail['dx_cognitive_load']:.4f}")

    st.divider()

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Commits", run_detail["total_commits"])
    s2.metric("Hotspot Files", run_detail["hotspot_file_count"])
    s3.metric("Coupling Pairs", run_detail["coupling_pair_count"])
    s4.metric("Knowledge Islands", run_detail["knowledge_island_count"])
    s5.metric("High Complexity", run_detail["complexity_high_count"])

# ── Tab 2: Hotspots ─────────────────────────────────────────────────

with tabs[1]:
    st.header("Hotspot Analysis")
    hotspots = fetch(f"/api/runs/{selected_run_id}/hotspots") or []
    if hotspots:
        st.dataframe(
            [{k: v for k, v in h.items() if k != "run_id"} for h in hotspots],
            use_container_width=True,
        )
        top = hotspots[:20]
        fig = go.Figure(go.Bar(
            x=[h["hotspot_score"] for h in top],
            y=[h["file_path"] for h in top],
            orientation="h",
            marker_color="#ff6b6b",
        ))
        fig.update_layout(
            title="Top 20 Hotspots",
            xaxis_title="Hotspot Score",
            yaxis=dict(autorange="reversed"),
            height=max(400, len(top) * 25),
            margin=dict(l=300),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hotspot data.")

# ── Tab 3: Knowledge ───────────────────────────────────────────────

with tabs[2]:
    st.header("Knowledge Distribution")

    k1, k2 = st.columns(2)
    k1.metric("Developer Risk Index", run_detail["developer_risk_index"])
    k2.metric("Knowledge Islands", run_detail["knowledge_island_count"])

    knowledge = fetch(f"/api/runs/{selected_run_id}/knowledge") or []
    if knowledge:
        st.dataframe(
            [{k: v for k, v in f.items() if k != "run_id"} for f in knowledge],
            use_container_width=True,
        )
    else:
        st.info("No knowledge data.")

# ── Tab 4: Coupling ─────────────────────────────────────────────────

with tabs[3]:
    st.header("Temporal Coupling & PAIN")

    coupling = fetch(f"/api/runs/{selected_run_id}/coupling") or []
    pain = fetch(f"/api/runs/{selected_run_id}/pain") or []

    if coupling:
        st.subheader("Coupling Pairs")
        st.dataframe(
            [{k: v for k, v in c.items() if k != "run_id"} for c in coupling],
            use_container_width=True,
        )

    if pain:
        st.subheader("PAIN Scores")
        st.dataframe(
            [{k: v for k, v in p.items() if k != "run_id"} for p in pain],
            use_container_width=True,
        )
        top_pain = pain[:20]
        fig = go.Figure(go.Bar(
            x=[p["pain_score"] for p in top_pain],
            y=[p["file_path"] for p in top_pain],
            orientation="h",
            marker_color="#ffa94d",
        ))
        fig.update_layout(
            title="Top 20 PAIN Scores",
            xaxis_title="PAIN Score",
            yaxis=dict(autorange="reversed"),
            height=max(400, len(top_pain) * 25),
            margin=dict(l=300),
        )
        st.plotly_chart(fig, use_container_width=True)

    if not coupling and not pain:
        st.info("No coupling/PAIN data.")

# ── Tab 5: Complexity ───────────────────────────────────────────────

with tabs[4]:
    st.header("Complexity Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Functions", run_detail["complexity_total_functions"])
    c2.metric("Avg Complexity", f"{run_detail['complexity_avg']:.2f}")
    c3.metric("Max Complexity", run_detail["complexity_max"])

    complexity = fetch(f"/api/runs/{selected_run_id}/complexity") or []
    if complexity:
        st.dataframe(
            [{k: v for k, v in f.items() if k != "run_id"} for f in complexity],
            use_container_width=True,
        )
        cc_values = [f["cyclomatic_complexity"] for f in complexity]
        fig = go.Figure(go.Histogram(
            x=cc_values,
            nbinsx=max(10, max(cc_values) if cc_values else 10),
            marker_color="#51cf66",
        ))
        fig.update_layout(
            title="Cyclomatic Complexity Distribution",
            xaxis_title="Cyclomatic Complexity",
            yaxis_title="Count",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No complexity data.")

# ── Tab 6: Clustering ──────────────────────────────────────────────

with tabs[5]:
    st.header("Change Clustering")

    cl1, cl2 = st.columns(2)
    cl1.metric("Clusters (k)", run_detail["clustering_k"])
    cl2.metric("Silhouette", f"{run_detail['clustering_silhouette']:.4f}")

    clusters = fetch(f"/api/runs/{selected_run_id}/clusters") or []
    drift = fetch(f"/api/runs/{selected_run_id}/drift") or []

    if clusters:
        fig = go.Figure(go.Pie(
            labels=[c["label"] for c in clusters],
            values=[c["size"] for c in clusters],
            hole=0.4,
        ))
        fig.update_layout(title="Cluster Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cluster Summaries")
        st.dataframe(
            [{k: v for k, v in c.items() if k != "run_id"} for c in clusters],
            use_container_width=True,
        )

    if drift:
        st.subheader("Cluster Drift")
        drift_display = []
        for d in drift:
            trend_arrow = {"growing": "+", "shrinking": "-", "stable": "="}
            arrow = trend_arrow.get(d["trend"], "")
            drift_display.append({
                "Label": d["cluster_label"],
                "1st Half %": f"{d['first_half_pct']:.1f}",
                "2nd Half %": f"{d['second_half_pct']:.1f}",
                "Drift": f"{d['drift']:+.1f}",
                "Trend": f"{arrow} {d['trend']}",
            })
        st.dataframe(drift_display, use_container_width=True)

    if not clusters and not drift:
        st.info("No clustering data.")

# ── Tab 7: Effort ──────────────────────────────────────────────────

with tabs[6]:
    st.header("Effort Modeling")

    e1, e2 = st.columns(2)
    e1.metric("Model R\u00b2", f"{run_detail['effort_model_r_squared']:.4f}")
    e2.metric("Total Files", run_detail["effort_total_files"])

    effort = fetch(f"/api/runs/{selected_run_id}/effort") or []
    if effort:
        st.dataframe(
            [{k: v for k, v in f.items() if k != "run_id"} for f in effort],
            use_container_width=True,
        )
        top_effort = effort[:20]
        fig = go.Figure(go.Bar(
            x=[f["rei_score"] for f in top_effort],
            y=[f["file_path"] for f in top_effort],
            orientation="h",
            marker_color="#748ffc",
        ))
        fig.update_layout(
            title="Top 20 REI Scores",
            xaxis_title="REI Score",
            yaxis=dict(autorange="reversed"),
            height=max(400, len(top_effort) * 25),
            margin=dict(l=300),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No effort data.")

# ── Tab 8: Anemia ──────────────────────────────────────────────────

with tabs[7]:
    st.header("Anemic Domain Model Detection")

    a1, a2, a3 = st.columns(3)
    a1.metric("Total Classes", run_detail["anemia_total_classes"])
    a2.metric("Anemic", f"{run_detail['anemia_anemic_count']} ({run_detail['anemia_anemic_pct']:.1f}%)")
    a3.metric("Avg AMS", f"{run_detail['anemia_average_ams']:.4f}")

    anemia = fetch(f"/api/runs/{selected_run_id}/anemia") or []
    if anemia:
        st.dataframe(
            [{k: v for k, v in c.items() if k != "run_id"} for c in anemia],
            use_container_width=True,
        )
    else:
        st.info("No anemia data.")

# ── Tab 9: Time Travel ─────────────────────────────────────────────

with tabs[8]:
    st.header("Time Travel / Run Comparison")

    if not compare or not compare_run_id:
        st.info("Enable comparison in the sidebar and select a second run.")
    else:
        comparison = fetch("/api/compare", params={"a": selected_run_id, "b": compare_run_id})
        if comparison is None:
            st.error("Comparison failed.")
        else:
            run_a = comparison["run_a"]
            run_b = comparison["run_b"]
            deltas = comparison["deltas"]

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader(f"Run A: {str(run_a['created_at'])[:19]}")
                st.metric("DX Score", f"{run_a['dx_score']:.4f}")
            with col_b:
                st.subheader(f"Run B: {str(run_b['created_at'])[:19]}")
                st.metric("DX Score", f"{run_b['dx_score']:.4f}",
                          delta=f"{deltas['dx_score']:+.4f}")

            st.divider()
            st.subheader("Metric Deltas")

            delta_cols = st.columns(3)
            delta_items = list(deltas.items())
            for i, (key, val) in enumerate(delta_items):
                col = delta_cols[i % 3]
                label = key.replace("_", " ").title()
                color = "normal" if key == "dx_cognitive_load" else "normal"
                col.metric(label, f"{getattr(run_b, key, val):.4f}" if isinstance(val, float) else str(val),
                           delta=f"{val:+.6f}" if val != 0 else "0")

            st.divider()

            # Side-by-side hotspots
            st.subheader("Hotspot Comparison")
            hotspots_a = fetch(f"/api/runs/{selected_run_id}/hotspots") or []
            hotspots_b = fetch(f"/api/runs/{compare_run_id}/hotspots") or []

            ha, hb = st.columns(2)
            with ha:
                st.caption("Run A Hotspots")
                if hotspots_a:
                    st.dataframe(
                        [{k: v for k, v in h.items() if k != "run_id"} for h in hotspots_a[:20]],
                        use_container_width=True,
                    )
            with hb:
                st.caption("Run B Hotspots")
                if hotspots_b:
                    st.dataframe(
                        [{k: v for k, v in h.items() if k != "run_id"} for h in hotspots_b[:20]],
                        use_container_width=True,
                    )
