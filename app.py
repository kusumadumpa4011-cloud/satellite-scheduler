
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Satellite Mission Control",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Intelligent Satellite Mission Control")
st.caption(
    "Satellite communication, task scheduling "
    "and resource monitoring"
)

st.sidebar.header("Mission Settings")

scenario = st.sidebar.selectbox(
    "Select mission scenario",
    [
        "Normal Mission",
        "Urgent Tasking",
        "Ground Station Outage",
        "Link Degradation",
        "Multiple Events"
    ]
)

st.sidebar.info(
    "This demo uses simulated mission data."
)

# Simulated mission summary
metrics = {
    "Normal Mission": (16, 2, 127),
    "Urgent Tasking": (16, 2, 157),
    "Ground Station Outage": (16, 2, 127),
    "Link Degradation": (16, 2, 127),
    "Multiple Events": (13, 5, 132)
}

scheduled, unscheduled, utility = metrics[scenario]

col1, col2, col3 = st.columns(3)

col1.metric("Tasks Scheduled", scheduled)
col2.metric("Tasks Unscheduled", unscheduled)
col3.metric("Mission Utility", utility)

if scenario in [
    "Ground Station Outage",
    "Link Degradation"
]:
    st.warning(
        "This scenario is currently illustrative. "
        "Actual outage or link degradation constraints "
        "are not yet simulated in this demo."
    )

st.divider()

st.subheader("Satellite Task Schedule")

schedule = pd.DataFrame({
    "Task": ["T01", "T02", "T03", "T04", "T05", "T06"],
    "Satellite": [
        "SAT-01", "SAT-02", "SAT-03",
        "SAT-01", "SAT-02", "SAT-03"
    ],
    "Start (min)": [0, 5, 10, 15, 20, 25],
    "Duration (min)": [5, 4, 6, 5, 3, 7],
    "Priority": [10, 8, 9, 7, 6, 10]
})

fig = px.bar(
    schedule,
    x="Duration (min)",
    y="Satellite",
    base="Start (min)",
    color="Satellite",
    hover_data=["Task", "Priority"],
    orientation="h",
    title="Illustrative Task Timeline"
)

fig.update_layout(
    xaxis_title="Mission Time (minutes)",
    yaxis_title="Satellite",
    legend_title="Satellite",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Task Details")
st.dataframe(schedule, use_container_width=True)

st.divider()

st.subheader("Resource Monitoring")

resources = pd.DataFrame({
    "Satellite": ["SAT-01", "SAT-02", "SAT-03"],
    "Power Used (%)": [65, 72, 58],
    "Storage Used (%)": [55, 68, 40]
})

fig2 = px.bar(
    resources,
    x="Satellite",
    y=["Power Used (%)", "Storage Used (%)"],
    barmode="group",
    title="Illustrative Resource Usage"
)

st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "Hackathon prototype | Simulated data, "
    "not live satellite telemetry. "
    "Illustrative charts are not optimizer outputs."
)

