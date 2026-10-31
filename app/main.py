import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 AI-Powered Data Analyst")
st.caption("Upload CSV files, ask questions in plain English, and explore your data.")

with st.sidebar:
    st.header("Data sources")
    uploads = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)
    if uploads and st.button("Load CSV files", type="primary"):
        files = [("files", (f.name, f.getvalue(), "text/csv")) for f in uploads]
        try:
            r = requests.post(f"{API_URL}/datasets/upload", files=files, timeout=60)
            r.raise_for_status()
            payload = r.json()
            st.session_state["dataset_id"] = payload["dataset_id"]
            st.session_state["dataset_info"] = payload["files"]
            st.session_state["messages"] = []
            st.success(f"Loaded {len(payload['files'])} file(s).")
        except requests.RequestException as exc:
            st.error(f"Backend unavailable or upload failed: {exc}")

    for info in st.session_state.get("dataset_info", []):
        st.caption(f"{info['name']} · {info['rows']} rows × {info['columns']} columns")

if not st.session_state.get("dataset_id"):
    st.info("Upload a CSV file from the sidebar to begin.")
    st.markdown("**Example questions:** Which region generated the highest revenue? · Show monthly sales trends · Detect anomalies · What columns are available?")
    st.stop()

dataset_id = st.session_state["dataset_id"]
tab_chat, tab_preview, tab_quality, tab_about = st.tabs(["💬 Ask your data", "🧾 Preview", "🧪 Data quality", "ℹ️ About"])

with tab_preview:
    try:
        r = requests.get(f"{API_URL}/datasets/{dataset_id}/preview", timeout=30)
        r.raise_for_status()
        for item in r.json()["files"]:
            st.subheader(item["name"])
            st.dataframe(pd.DataFrame(item["preview"]), width="stretch")
    except requests.RequestException as exc:
        st.error(f"Preview failed: {exc}")

with tab_quality:
    try:
        r = requests.get(f"{API_URL}/datasets/{dataset_id}/quality", timeout=30)
        r.raise_for_status()
        for item in r.json()["files"]:
            st.subheader(item["name"])
            c1,c2,c3 = st.columns(3)
            c1.metric("Rows", item["rows"])
            c2.metric("Columns", item["columns"])
            c3.metric("Duplicate rows", item["duplicate_rows"])
            st.write("Missing values")
            st.json(item["missing_values"])
            st.write("Column data types")
            st.json(item["dtypes"])
    except requests.RequestException as exc:
        st.error(f"Quality check failed: {exc}")

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("table") is not None:
                st.dataframe(pd.DataFrame(msg["table"]), width="stretch")
            if msg.get("chart"):
                spec = msg["chart"]
                frame = pd.DataFrame(spec["data"])
                x, y, kind = spec["x"], spec["y"], spec["type"]
                if x in frame.columns and y in frame.columns:
                    if kind == "line": fig = px.line(frame, x=x, y=y, title=spec.get("title"))
                    elif kind == "pie": fig = px.pie(frame, names=x, values=y, title=spec.get("title"))
                    elif kind == "scatter": fig = px.scatter(frame, x=x, y=y, title=spec.get("title"))
                    else: fig = px.bar(frame, x=x, y=y, title=spec.get("title"))
                    st.plotly_chart(fig, width="stretch")
            if msg.get("sql"):
                with st.expander("Generated SQL"): st.code(msg["sql"], language="sql")
            if msg.get("pandas_code"):
                with st.expander("Pandas code"): st.code(msg["pandas_code"], language="python")
    question = st.chat_input("Ask a question about your data...")
    if question:
        st.session_state["messages"].append({"role":"user","content":question})
        with st.chat_message("user"): st.markdown(question)
        history = [{"role":m["role"],"content":m["content"]} for m in st.session_state["messages"][-8:]]
        try:
            r = requests.post(f"{API_URL}/analyze", json={"dataset_id":dataset_id,"question":question,"history":history}, timeout=120)
            r.raise_for_status()
            answer = r.json()
            message = {
                "role":"assistant", "content":answer["answer"], "table":answer.get("table"),
                "chart":answer.get("chart"), "sql":answer.get("sql"), "pandas_code":answer.get("pandas_code")
            }
            st.session_state["messages"].append(message)
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Analysis request failed: {exc}")

with tab_about:
    st.markdown("""
    ### How it works
    1. Upload one or more CSV files.
    2. Inspect previews and data-quality checks.
    3. Ask a question in natural language.
    4. The backend selects a constrained analysis plan and computes results with Pandas.

    ### Safety
    The app does not execute arbitrary Python or SQL generated by an LLM. SQL is illustrative only.
    """)
