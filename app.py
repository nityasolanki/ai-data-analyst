import streamlit as st
import pandas as pd
import os
import io
from dotenv import load_dotenv
from agent import DataAnalystAgent

load_dotenv()
from utils import (generate_chart, build_dashboard, df_to_csv_bytes, fig_to_png_bytes,
                   load_google_sheet, detect_anomalies, build_anomaly_chart,
                   predict_trend, generate_pdf_report)

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0d0f14; }
.main-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
    border: 1px solid #2a3441; border-radius: 16px;
    padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.main-header::before {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
}
.main-header h1 { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; margin: 0 0 0.2rem; }
.main-header p { color: #64748b; font-size: 0.9rem; margin: 0; }
.badge {
    display: inline-block; background: rgba(99,179,237,0.1); color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.2); border-radius: 20px;
    padding: 2px 10px; font-size: 0.72rem; font-weight: 500;
    margin-bottom: 0.6rem; letter-spacing: 0.5px; text-transform: uppercase;
}
.stat-card {
    background: #1a1f2e; border: 1px solid #2a3441;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
}
.stat-label { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-value { color: #e2e8f0; font-size: 1.3rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.chat-user {
    background: linear-gradient(135deg, #1e3a5f, #1a2d4a);
    border: 1px solid #2a4a6b; border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.1rem; margin: 0.5rem 0; color: #bfdbfe; font-size: 0.95rem;
}
.chat-assistant {
    background: #1a1f2e; border: 1px solid #2a3441;
    border-radius: 12px 12px 12px 4px;
    padding: 0.9rem 1.1rem; margin: 0.5rem 0; color: #cbd5e1;
    font-size: 0.95rem; line-height: 1.6;
}
.chat-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; font-weight: 600; }
.user-label { color: #63b3ed; }
.agent-label { color: #68d391; }
.profile-card {
    background: linear-gradient(135deg, rgba(104,211,145,0.05), rgba(99,179,237,0.05));
    border: 1px solid rgba(104,211,145,0.15); border-left: 3px solid #68d391;
    border-radius: 10px; padding: 1rem 1.2rem; margin: 0.5rem 0;
    color: #a7f3d0; font-size: 0.88rem; line-height: 1.7;
}
.insight-box {
    background: rgba(99,179,237,0.04); border: 1px solid rgba(99,179,237,0.15);
    border-radius: 10px; padding: 1rem 1.2rem; margin: 0.5rem 0;
    color: #93c5fd; font-size: 0.9rem;
}
.memory-badge {
    display: inline-block; background: rgba(168,85,247,0.1); color: #c084fc;
    border: 1px solid rgba(168,85,247,0.2); border-radius: 6px;
    padding: 1px 8px; font-size: 0.7rem; margin-left: 8px; vertical-align: middle;
}
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #1e40af) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
}
[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2530 !important; }
.stDataFrame { border-radius: 10px; overflow: hidden; }
div[data-testid="stExpander"] { background: #1a1f2e; border: 1px solid #2a3441; border-radius: 10px; }
.stTabs [data-baseweb="tab"] { color: #64748b; font-family: 'Space Grotesk', sans-serif; }
.stTabs [aria-selected="true"] { color: #63b3ed !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [("messages", []), ("agent", None), ("df", None),
                 ("profile", None), ("dashboard_fig", None), ("file_name", "")]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    groq_key = st.text_input("Groq API Key", type="password",
                             value=os.getenv("GROQ_API_KEY", ""),
                             help="Free at console.groq.com")

    st.markdown("---")
    st.markdown("### 📂 Load Data")

    data_source = st.radio("Source", ["Upload CSV", "Google Sheets URL"], horizontal=True)

    df_loaded = None

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Drop a CSV file", type=["csv", "xlsx"])
        if uploaded:
            try:
                if uploaded.name.endswith(".xlsx"):
                    df_loaded = pd.read_excel(uploaded)
                else:
                    df_loaded = pd.read_csv(uploaded)
                st.session_state.file_name = uploaded.name
            except Exception as e:
                st.error(f"Error: {e}")

    else:
        sheet_url = st.text_input("Public Google Sheet URL",
                                  placeholder="https://docs.google.com/spreadsheets/d/...")
        if st.button("Load Sheet", use_container_width=True) and sheet_url:
            try:
                with st.spinner("Fetching sheet..."):
                    df_loaded = load_google_sheet(sheet_url)
                st.session_state.file_name = "Google Sheet"
            except Exception as e:
                st.error(f"Could not load sheet: {e}. Make sure sharing is set to 'Anyone with link'.")

    if df_loaded is not None:
        st.session_state.df = df_loaded
        st.session_state.messages = []  # reset chat on new file
        if groq_key:
            st.session_state.agent = DataAnalystAgent(df_loaded, groq_key)
            with st.spinner("Auto-profiling data..."):
                st.session_state.profile = st.session_state.agent.get_auto_profile()
                st.session_state.dashboard_fig = build_dashboard(df_loaded)
        st.success(f"✅ Loaded {len(df_loaded):,} rows × {len(df_loaded.columns)} cols")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.markdown("### 📊 Dataset")
        st.markdown(f"""
        <div class="stat-card"><div class="stat-label">Rows</div><div class="stat-value">{len(df):,}</div></div>
        <div class="stat-card"><div class="stat-label">Columns</div><div class="stat-value">{len(df.columns)}</div></div>
        <div class="stat-card"><div class="stat-label">Missing</div><div class="stat-value">{df.isnull().sum().sum():,}</div></div>
        """, unsafe_allow_html=True)

        # Memory indicator
        if st.session_state.agent:
            turns = len(st.session_state.agent.history) // 2
            st.markdown(f"**Memory** <span class='memory-badge'>{turns} turns</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Try asking...")
    examples = [
        "Show a bar chart of revenue by region",
        "Who are the top 5 sales reps by profit?",
        "Now filter that to just the North region",
        "Is there any correlation in this data?",
        "Give me a full statistical summary",
        "Are there any missing values?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.pending_question = ex

    st.markdown("---")
    if st.session_state.df is not None and st.session_state.messages and st.session_state.profile:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Building report…"):
                pdf = generate_pdf_report(
                    st.session_state.df,
                    st.session_state.messages,
                    st.session_state.profile
                )
            st.download_button("⬇️ Download Report (PDF)", pdf,
                               "analysis_report.pdf", "application/pdf",
                               use_container_width=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.agent:
            st.session_state.agent.history = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="badge">🤖 LLaMA 3.3 · Groq · Pandas · Memory</div>
    <h1>📊 AI Data Analyst Agent</h1>
    <p>Upload CSV or paste a Google Sheet · Ask follow-up questions · Export charts & tables</p>
</div>
""", unsafe_allow_html=True)

# ── Process queued input BEFORE rendering tabs (fixes vanishing message bug) ──
if "pending_question" in st.session_state:
    _q = st.session_state.pop("pending_question")
    if _q and st.session_state.df is not None and groq_key:
        st.session_state["_queued_input"] = _q

if "_queued_input" in st.session_state:
    _input = st.session_state.pop("_queued_input")
    if st.session_state.agent and _input:
        st.session_state.messages.append({"role": "user", "content": _input})
        with st.spinner("🧠 Agent thinking…"):
            try:
                result = st.session_state.agent.run(_input)
                _msg = {"role": "assistant", "content": result["text"]}
                if result.get("chart"):
                    _msg["chart"] = result["chart"]
                if result.get("dataframe") is not None:
                    _msg["dataframe"] = result["dataframe"]
                st.session_state.messages.append(_msg)
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {str(e)}"
                })

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_dashboard, tab_profile, tab_data, tab_anomaly, tab_predict = st.tabs([
    "💬 Chat", "📊 Dashboard", "🔍 Auto Profile", "🗃️ Data", "🚨 Anomalies", "🔮 Predict"
])

# ═══════════════════════ TAB 1: CHAT ═════════════════════════════════════════
with tab_chat:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Chat history
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-user">
                    <div class="chat-label user-label">You</div>
                    {msg["content"]}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-assistant">
                    <div class="chat-label agent-label">🤖 Agent</div>
                    {msg["content"]}
                </div>""", unsafe_allow_html=True)

                # Chart with export button
                if "chart" in msg:
                    st.plotly_chart(msg["chart"], use_container_width=True, key=f"chart_{i}")
                    try:
                        png = fig_to_png_bytes(msg["chart"])
                        st.download_button(
                            "⬇️ Download Chart (PNG)", png,
                            file_name=f"chart_{i}.png", mime="image/png",
                            key=f"dl_chart_{i}"
                        )
                    except Exception:
                        pass

                # Table with export button
                if "dataframe" in msg:
                    st.dataframe(msg["dataframe"], use_container_width=True, key=f"df_{i}")
                    csv_bytes = df_to_csv_bytes(msg["dataframe"])
                    st.download_button(
                        "⬇️ Download Table (CSV)", csv_bytes,
                        file_name=f"result_{i}.csv", mime="text/csv",
                        key=f"dl_df_{i}"
                    )

        # Chat input — queues to session state, processed at top of script
        st.markdown("---")
        user_input = st.chat_input("Ask anything about your data… (follow-ups work too!)")

        if user_input:
            if st.session_state.df is None:
                st.warning("⬅️ Please upload a CSV or connect a Google Sheet first.")
            elif not groq_key:
                st.warning("⬅️ Please enter your Groq API key in the sidebar.")
            else:
                st.session_state["_queued_input"] = user_input
                st.rerun()

        if not st.session_state.messages:
            st.markdown("""
            <div class="insight-box">
                👈 Upload a file or connect a Google Sheet, then start asking questions.<br><br>
                The agent <strong>remembers your conversation</strong> — ask follow-ups like<br>
                "now filter that by region" or "show it as a pie chart instead".
            </div>
            """, unsafe_allow_html=True)

    with col2:
        if st.session_state.df is not None:
            df = st.session_state.df
            st.markdown("#### 📋 Columns")
            col_info = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Nulls": df.isnull().sum().values,
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)

            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                st.markdown("#### 📈 Quick Stats")
                st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

# ═══════════════════════ TAB 2: DASHBOARD ════════════════════════════════════
with tab_dashboard:
    if st.session_state.dashboard_fig:
        st.plotly_chart(st.session_state.dashboard_fig, use_container_width=True)
        try:
            png = fig_to_png_bytes(st.session_state.dashboard_fig)
            st.download_button("⬇️ Download Dashboard (PNG)", png,
                               file_name="dashboard.png", mime="image/png")
        except Exception:
            pass
    elif st.session_state.df is not None and not groq_key:
        st.info("Enter your Groq API key to generate the auto-dashboard.")
    else:
        st.markdown("""
        <div class="insight-box">
            📊 Upload a dataset to auto-generate a 4-panel dashboard.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════ TAB 3: AUTO PROFILE ═════════════════════════════════
with tab_profile:
    if st.session_state.profile:
        p = st.session_state.profile
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{p['shape']['rows']:,}")
        c2.metric("Columns", f"{p['shape']['cols']}")
        c3.metric("Missing Values", f"{p['missing_total']:,}")
        c4.metric("Duplicate Rows", f"{p['duplicate_rows']:,}")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 🔢 Numeric KPI Columns")
            if p.get("kpi_cols"):
                for kpi in p["kpi_cols"]:
                    st.markdown(f"- `{kpi}`")
            else:
                st.markdown("No KPI columns detected.")

            if p.get("missing_cols"):
                st.markdown("#### ⚠️ Columns With Missing Data")
                missing_df = pd.DataFrame(
                    [(k, v) for k, v in p["missing_cols"].items()],
                    columns=["Column", "Missing Count"]
                )
                st.dataframe(missing_df, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("#### 🏷️ Categorical Breakdown")
            for col, info in p.get("top_categoricals", {}).items():
                st.markdown(f"""
                <div class="profile-card">
                    <strong>{col}</strong><br>
                    {info['unique']} unique values · Top: <code>{info['top']}</code> ({info['top_count']} times)
                </div>
                """, unsafe_allow_html=True)

        if p.get("most_variable_col"):
            st.markdown(f"**Most variable column:** `{p['most_variable_col']}`")
    else:
        st.markdown("""
        <div class="insight-box">
            🔍 Upload a dataset to see the automatic data profile report.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════ TAB 4: RAW DATA ═════════════════════════════════════
with tab_data:
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"**{st.session_state.file_name}** — {len(df):,} rows × {len(df.columns)} columns")

        # Download full dataset
        st.download_button(
            "⬇️ Download Full Dataset (CSV)",
            df_to_csv_bytes(df),
            file_name=st.session_state.file_name.replace(".xlsx", ".csv"),
            mime="text/csv"
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.markdown("""
        <div class="insight-box">
            🗃️ Upload a dataset to view the raw data here.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════ TAB 5: ANOMALIES ════════════════════════════════════
with tab_anomaly:
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("### 🚨 Anomaly & Outlier Detection")
        st.markdown("Outliers detected using the **IQR method** across all numeric columns.")

        anomalies = detect_anomalies(df)

        if not anomalies:
            st.success("✅ No outliers detected in any numeric column.")
        else:
            for col, info in anomalies.items():
                with st.expander(f"⚠️ `{col}` — {info['count']} outliers found", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Outlier Rows", info["count"])
                    c2.metric("Lower Bound", info["lower_bound"])
                    c3.metric("Upper Bound", info["upper_bound"])
                    st.plotly_chart(build_anomaly_chart(df, col), use_container_width=True)
                    st.dataframe(info["outlier_rows"], use_container_width=True)
                    csv = df_to_csv_bytes(info["outlier_rows"])
                    st.download_button(f"⬇️ Download {col} outliers",
                                       csv, f"{col}_outliers.csv",
                                       key=f"dl_anomaly_{col}")
    else:
        st.markdown('<div class="insight-box">Upload a dataset to run anomaly detection.</div>',
                    unsafe_allow_html=True)

# ═══════════════════════ TAB 6: PREDICT ══════════════════════════════════════
with tab_predict:
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("### 🔮 Predictive Trend Analysis")
        st.markdown("Fits a **linear regression** on any numeric column and forecasts ahead.")

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.warning("No numeric columns found.")
        else:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                target = st.selectbox("Select column to forecast", num_cols)
            with col_b:
                periods = st.slider("Forecast periods", 3, 20, 5)

            if st.button("🔮 Run Forecast", use_container_width=True):
                with st.spinner("Fitting model…"):
                    result = predict_trend(df, target, periods)

                st.plotly_chart(result["chart"], use_container_width=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("R² Score", result["r2_score"],
                          help="1.0 = perfect fit, 0 = no fit")
                m2.metric("Trend Direction", result["direction"])
                m3.metric("Slope per period", result["slope"])

                st.markdown("#### Forecasted Values")
                forecast_df = pd.DataFrame({
                    "Period": [f"+{i+1}" for i in range(periods)],
                    f"Predicted {target}": result["forecast_values"]
                })
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download Forecast",
                                   df_to_csv_bytes(forecast_df),
                                   f"{target}_forecast.csv")

                try:
                    png = fig_to_png_bytes(result["chart"])
                    st.download_button("⬇️ Download Chart (PNG)", png,
                                       f"{target}_forecast.png", "image/png")
                except Exception:
                    pass
    else:
        st.markdown('<div class="insight-box">Upload a dataset to run predictions.</div>',
                    unsafe_allow_html=True)