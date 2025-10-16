import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime, timedelta

st.set_page_config(page_title="Jeffrey OS Dashboard", layout="wide")

API_BASE = "http://localhost:8000"

st.title("🧠 Jeffrey OS - Brain Dashboard")

# Sidebar pour les contrôles
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🔄 Refresh Data"):
        st.rerun()

    st.subheader("Dream Engine")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Run Dream"):
            response = requests.post(f"{API_BASE}/api/v1/dream/run?force=true")
            if response.status_code == 200:
                st.success("Dream cycle started!")

    with col2:
        if st.button("📊 Dream Status"):
            response = requests.get(f"{API_BASE}/api/v1/dream/status")
            st.json(response.json())

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["💭 Emotion Detection", "📚 Memory Timeline", "🎯 Metrics", "🔍 Search"])

with tab1:
    st.header("Emotion Detection")
    text = st.text_area("Enter text for analysis:", height=100)

    if st.button("Detect Emotion", disabled=not text):
        try:
            response = requests.get(f"{API_BASE}/api/v1/emotion/health")

            if response.status_code == 200:
                result = response.json()
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Test Emotion", result['test_emotion'])
                    st.metric("Confidence", f"{result['test_confidence']:.1%}")

                with col2:
                    st.write("Method:", result['method'])
                    st.write("Status:", result['status'])

            else:
                st.error("Emotion detection service unavailable")
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.header("Memory Timeline")

    hours = st.slider("Hours to look back:", 1, 168, 24)
    limit = st.slider("Max memories:", 10, 100, 50)

    try:
        response = requests.get(f"{API_BASE}/api/v1/memories/recent?hours={hours}&limit={limit}")

        if response.status_code == 200:
            memories = response.json()

            if memories:
                st.write(f"Found {len(memories)} memories")

                for mem in memories[:20]:  # Show max 20
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(mem['text'])
                        with col2:
                            if mem.get('emotion'):
                                st.caption(f"😊 {mem['emotion']}")
                        with col3:
                            if mem.get('processed'):
                                st.caption("✅ Processed")
                            else:
                                st.caption("⏳ Pending")

                        st.caption(f"🕐 {mem['timestamp']}")
                        st.divider()
            else:
                st.info("No memories found")
        else:
            st.error("Failed to load memories")
    except Exception as e:
        st.error(f"Error loading memories: {e}")

with tab3:
    st.header("System Metrics")

    try:
        response = requests.get(f"{API_BASE}/metrics")
        if response.status_code == 200:
            metrics_text = response.text

            # Parse Jeffrey metrics
            jeffrey_metrics = [line for line in metrics_text.split('\n')
                              if 'jeffrey_' in line and not line.startswith('#')]

            # Display key metrics
            col1, col2, col3 = st.columns(3)

            for metric in jeffrey_metrics[:9]:
                if 'dream_quality' in metric:
                    value = metric.split()[-1]
                    col1.metric("Dream Quality", value)
                elif 'dream_batch_size' in metric:
                    value = metric.split()[-1]
                    col2.metric("Batch Size", value)
                elif 'emotion_requests_total' in metric:
                    value = metric.split()[-1]
                    col3.metric("Total Emotions", value)

            # Show raw metrics
            with st.expander("Raw Metrics"):
                st.text('\n'.join(jeffrey_metrics))
        else:
            st.error("Failed to load metrics")
    except Exception as e:
        st.error(f"Error loading metrics: {e}")

with tab4:
    st.header("Memory Search")

    search_query = st.text_input("Search memories:")

    if search_query and st.button("Search"):
        try:
            response = requests.get(
                f"{API_BASE}/api/v1/memories/search",
                params={"query": search_query, "limit": 20}
            )

            if response.status_code == 200:
                results = response.json()
                st.write(f"Found {len(results)} matches")

                for mem in results:
                    st.write(f"- {mem['text']}")
                    st.caption(f"  Emotion: {mem.get('emotion', 'N/A')} | Time: {mem['timestamp']}")
            else:
                st.error("Search failed")
        except Exception as e:
            st.error(f"Search error: {e}")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Health Check"):
        try:
            response = requests.get(f"{API_BASE}/healthz")
            if response.status_code == 200:
                st.success("✅ API Healthy")
            else:
                st.error("❌ API Unhealthy")
        except:
            st.error("❌ API Unreachable")

with col2:
    if st.button("Ready Check"):
        try:
            response = requests.get(f"{API_BASE}/readyz")
            if response.status_code == 200:
                result = response.json()
                if result.get('ready'):
                    st.success("✅ System Ready")
                else:
                    st.warning("⚠️ System Not Ready")
        except:
            st.error("❌ Readiness Check Failed")

with col3:
    st.caption("Jeffrey OS - Production-Ready Brain Infrastructure")