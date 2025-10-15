"""
Jeffrey OS Dashboard - Streamlit Application
Phase 2.3 implementation
"""

import asyncio
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from jeffrey.core.loops.loop_manager import LoopManager
from jeffrey.utils.test_helpers import DummyMemoryFederation, NullBus, SimpleState

st.set_page_config(page_title="Jeffrey OS Dashboard", page_icon="🧠", layout="wide")

# CSS custom
st.markdown(
    """
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
}
.stAlert {
    background: rgba(102, 126, 234, 0.1);
    border-left: 3px solid #667eea;
}
</style>
""",
    unsafe_allow_html=True,
)


class JeffreyDashboard:
    """Dashboard principal pour Jeffrey OS"""

    def __init__(self):
        self.manager = None
        self.init_connection()

    def init_connection(self):
        """Connexion au LoopManager"""
        if "manager" not in st.session_state:
            # Create manager with test infrastructure
            manager = LoopManager(
                cognitive_core=SimpleState(),
                emotion_orchestrator=None,
                memory_federation=DummyMemoryFederation(),
                bus=NullBus(),
                mode_getter=lambda: "normal",
                latency_budget_ok=lambda: True,
            )
            st.session_state.manager = manager
            st.session_state.manager_started = False

        self.manager = st.session_state.manager

    async def ensure_manager_running(self):
        """Ensure manager is running"""
        if not st.session_state.manager_started:
            await self.manager.start()
            st.session_state.manager_started = True

    def render(self):
        """Render principal"""
        st.title("🧠 Jeffrey OS - Live Dashboard")

        # Sidebar controls
        with st.sidebar:
            st.header("⚙️ Controls")

            if st.button("🚀 Start Loops" if not st.session_state.get("manager_started") else "🛑 Stop Loops"):
                if not st.session_state.get("manager_started"):
                    asyncio.run(self.ensure_manager_running())
                    st.success("Loops started!")
                else:
                    asyncio.run(self.manager.stop())
                    st.session_state.manager_started = False
                    st.info("Loops stopped")
                st.rerun()

            st.divider()
            st.metric("Status", "🟢 Running" if st.session_state.get("manager_started") else "🔴 Stopped")

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Overview", "🔄 Loops", "🧠 Q-Learning", "🌐 Symbiosis", "💾 Memory"]
        )

        with tab1:
            self.render_overview()

        with tab2:
            self.render_loops()

        with tab3:
            self.render_qlearning()

        with tab4:
            self.render_symbiosis()

        with tab5:
            self.render_memory()

    def render_overview(self):
        """Vue d'ensemble"""
        st.header("System Overview")

        metrics = self.manager.get_metrics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Cycles",
                metrics["system"].get("total_cycles", 0),
                delta=f"+{metrics['system'].get('total_cycles', 0) // max(metrics['system'].get('uptime', 1), 1)}/s"
                if metrics["system"].get("uptime", 0) > 0
                else "0/s",
            )

        with col2:
            st.metric(
                "Bus Dropped",
                metrics["system"].get("bus_dropped", 0),
                delta=None
                if metrics["system"].get("bus_dropped", 0) == 0
                else f"+{metrics['system'].get('bus_dropped', 0)}",
            )

        with col3:
            st.metric("Symbiosis Score", f"{metrics['system'].get('symbiosis_score', 0):.3f}", delta=None)

        with col4:
            uptime = metrics["system"].get("uptime", 0)
            if uptime < 60:
                uptime_str = f"{uptime:.0f}s"
            elif uptime < 3600:
                uptime_str = f"{uptime / 60:.1f}m"
            else:
                uptime_str = f"{uptime / 3600:.1f}h"
            st.metric("Uptime", uptime_str)

        # System health
        st.divider()
        st.subheader("System Health")

        health_cols = st.columns(3)
        with health_cols[0]:
            error_rate = metrics["system"].get("total_errors", 0) / max(metrics["system"].get("total_cycles", 1), 1)
            st.metric("Error Rate", f"{error_rate:.2%}")

        with health_cols[1]:
            avg_latency = np.mean([loop.get("avg_latency_ms", 0) for loop in metrics.get("loops", {}).values()])
            st.metric("Avg Latency", f"{avg_latency:.1f}ms")

        with health_cols[2]:
            active_loops = sum(1 for loop in metrics.get("loops", {}).values() if loop.get("running", False))
            st.metric("Active Loops", active_loops)

    def render_loops(self):
        """État des boucles"""
        st.header("Loop Status")

        metrics = self.manager.get_metrics()

        # Table des loops
        loop_data = []
        for name, data in metrics.get("loops", {}).items():
            loop_data.append(
                {
                    "Loop": name.replace("_", " ").title(),
                    "Status": "🟢" if data.get("running", False) else "🔴",
                    "Cycles": data.get("cycles", 0),
                    "Errors": data.get("errors", 0),
                    "Avg Latency": f"{data.get('avg_latency_ms', 0):.1f}ms",
                    "P95 Latency": f"{data.get('p95_latency_ms', 0):.1f}ms",
                    "P99 Latency": f"{data.get('p99_latency_ms', 0):.1f}ms",
                }
            )

        if loop_data:
            df = pd.DataFrame(loop_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Graphique latences
            st.subheader("Latency Distribution")

            fig = go.Figure()

            for name, data in metrics.get("loops", {}).items():
                fig.add_trace(
                    go.Bar(
                        name=name.replace("_", " ").title(),
                        x=["P50", "P95", "P99"],
                        y=[
                            data.get("p50_latency_ms", 0),
                            data.get("p95_latency_ms", 0),
                            data.get("p99_latency_ms", 0),
                        ],
                    )
                )

            fig.update_layout(
                title="Latency Percentiles by Loop",
                xaxis_title="Percentile",
                yaxis_title="Latency (ms)",
                barmode="group",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No loop data available. Start the loops to see metrics.")

    def render_qlearning(self):
        """Visualisation Q-Learning"""
        st.header("Q-Learning Status")

        metrics = self.manager.get_metrics()

        # Find a loop with Q-table
        q_table_data = None
        replay_buffer_stats = None

        for name, loop in self.manager.loops.items():
            if hasattr(loop, "q_table") and loop.q_table:
                q_table_data = loop.q_table
                if hasattr(loop, "replay_buffer") and loop.replay_buffer:
                    replay_buffer_stats = loop.replay_buffer.get_stats()
                break

        if q_table_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Q-Table Size", len(q_table_data))
            with col2:
                if replay_buffer_stats:
                    st.metric(
                        "Replay Buffer",
                        f"{replay_buffer_stats['size']}/{replay_buffer_stats['capacity']}",
                    )
            with col3:
                total_values = sum(len(actions) for actions in q_table_data.values())
                st.metric("Total Q-Values", total_values)

            # Q-Table Heatmap
            st.subheader("Q-Values Heatmap")

            # Convert Q-table to DataFrame for visualization
            states = []
            actions = []
            values = []

            for state, action_values in list(q_table_data.items())[:50]:  # Limit to 50
                for action, value in action_values.items():
                    states.append(state[:30])  # Truncate state names
                    actions.append(action)
                    values.append(value)

            if values:
                # Create matrix for heatmap
                unique_states = list(set(states))
                unique_actions = list(set(actions))

                matrix = np.zeros((len(unique_states), len(unique_actions)))
                for i, (s, a, v) in enumerate(zip(states, actions, values)):
                    si = unique_states.index(s)
                    ai = unique_actions.index(a)
                    matrix[si, ai] = v

                fig = go.Figure(
                    data=go.Heatmap(
                        z=matrix,
                        x=unique_actions,
                        y=unique_states,
                        colorscale="Viridis",
                        text=matrix.round(2),
                        texttemplate="%{text}",
                        textfont={"size": 10},
                    )
                )

                fig.update_layout(
                    title="Q-Values by State and Action",
                    xaxis_title="Actions",
                    yaxis_title="States",
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Q-Learning data available yet. The system needs to run for a while to collect data.")

    def render_symbiosis(self):
        """Graphe de symbiose"""
        st.header("Symbiosis Analysis")

        metrics = self.manager.get_metrics()
        symbiosis_score = metrics["system"].get("symbiosis_score", 0)

        # Score gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=symbiosis_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Symbiosis Score"},
                delta={"reference": 0.7, "position": "bottom"},
                gauge={
                    "axis": {"range": [None, 1]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 0.3], "color": "lightgray"},
                        {"range": [0.3, 0.7], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 0.9,
                    },
                },
            )
        )

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Loop interaction graph
        if self.manager.symbiotic_graph and hasattr(self.manager.symbiotic_graph, "graph"):
            st.subheader("Loop Interactions")

            # Get graph data
            graph_analysis = asyncio.run(self.manager.symbiotic_graph.analyze_interactions())

            if "error" not in graph_analysis:
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Graph Nodes", graph_analysis["graph_metrics"]["nodes"])
                    st.metric("Graph Edges", graph_analysis["graph_metrics"]["edges"])

                with col2:
                    st.metric("Graph Density", f"{graph_analysis['graph_metrics']['density']:.3f}")
                    st.metric("Synergies", len(graph_analysis.get("synergies", [])))

                # Recommendations
                if graph_analysis.get("recommendations"):
                    st.subheader("Recommendations")
                    for rec in graph_analysis["recommendations"]:
                        st.info(rec)

    def render_memory(self):
        """Memory consolidation status"""
        st.header("Memory Consolidation")

        if "memory_consolidation" in self.manager.loops:
            loop = self.manager.loops["memory_consolidation"]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Memories Processed", getattr(loop, "memories_processed", 0))
            with col2:
                st.metric("Memories Archived", getattr(loop, "memories_archived", 0))
            with col3:
                st.metric("Memories Pruned", getattr(loop, "memories_pruned", 0))
            with col4:
                st.metric("Compressions", getattr(loop, "compression_count", 0))

            # Memory usage
            if hasattr(loop, "_get_memory_usage"):
                memory_usage = loop._get_memory_usage()
                st.metric("Memory Usage", f"{memory_usage:.2f} MB")

            # Compression stats
            if hasattr(loop, "compressed_history") and loop.compressed_history:
                st.subheader("Compression History")

                compression_data = pd.DataFrame(
                    [
                        {
                            "Time": datetime.fromtimestamp(item["timestamp"]).strftime("%H:%M:%S"),
                            "Count": item["count"],
                            "Ratio": f"{item['size_ratio']:.2%}",
                        }
                        for item in loop.compressed_history[-10:]  # Last 10
                    ]
                )

                if not compression_data.empty:
                    st.dataframe(compression_data, use_container_width=True, hide_index=True)
        else:
            st.info("Memory consolidation loop not available")


# Main entry point
def main():
    dashboard = JeffreyDashboard()

    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh (1s)", value=False)

    if auto_refresh:
        st_autorefresh = st.empty()
        dashboard.render()
        time.sleep(1)
        st.rerun()
    else:
        dashboard.render()


if __name__ == "__main__":
    main()
