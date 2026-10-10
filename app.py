
import math
import pandas as pd
import plotly.express as px
import streamlit as st
from ortools.sat.python import cp_model

st.set_page_config(
    page_title="Satellite Mission Control",
    page_icon="🛰️",
    layout="wide"
)

# ---------------- MISSION DATA ----------------

SATELLITES = {
    "SAT-01": {
        "power": 100, "storage": 120, "initial": 20,
        "capabilities": ["IMAGING", "COMMUNICATION"]
    },
    "SAT-02": {
        "power": 100, "storage": 120, "initial": 15,
        "capabilities": ["IMAGING"]
    },
    "SAT-03": {
        "power": 100, "storage": 120, "initial": 10,
        "capabilities": ["COMMUNICATION", "WEATHER"]
    }
}

STATIONS = {
    "GS-01": 20,
    "GS-02": 15,
    "GS-03": 10
}

PASSES = [
    ("P01", "SAT-01", "GS-01", 0, 10, .95),
    ("P02", "SAT-02", "GS-02", 5, 15, .90),
    ("P03", "SAT-03", "GS-03", 10, 20, .85),
    ("P04", "SAT-01", "GS-02", 15, 25, .80),
    ("P05", "SAT-02", "GS-01", 20, 30, .95),
    ("P06", "SAT-03", "GS-02", 25, 35, .75),
    ("P07", "SAT-01", "GS-03", 30, 40, .90),
    ("P08", "SAT-02", "GS-03", 35, 45, .85),
    ("P09", "SAT-03", "GS-01", 40, 50, .95),
    ("P10", "SAT-01", "GS-01", 45, 55, .70),
    ("P11", "SAT-02", "GS-02", 50, 60, .90),
    ("P12", "SAT-03", "GS-03", 55, 65, .80)
]

TASKS = [
    ("T01","IMAGING",10,5,25,20,15),
    ("T02","COMMUNICATION",8,4,15,25,10),
    ("T03","WEATHER",9,6,20,30,12),
    ("T04","IMAGING",7,5,30,35,18),
    ("T05","COMMUNICATION",6,3,10,40,8),
    ("T06","IMAGING",10,7,35,45,20),
    ("T07","WEATHER",5,4,12,50,9),
    ("T08","COMMUNICATION",8,5,18,55,11),
    ("T09","IMAGING",9,6,28,60,16),
    ("T10","WEATHER",7,3,10,65,7),
    ("T11","COMMUNICATION",6,4,16,55,10),
    ("T12","IMAGING",8,5,22,60,14),
    ("T13","WEATHER",9,4,14,50,9),
    ("T14","COMMUNICATION",7,6,24,65,13),
    ("T15","IMAGING",5,3,12,40,8),
    ("T16","WEATHER",10,5,26,60,15),
    ("T17","COMMUNICATION",9,4,20,45,12),
    ("T18","IMAGING",6,6,18,65,11)
]

# ---------------- OPTIMIZER ----------------

def optimize_mission(scenario, outage_station, degradation):
    model = cp_model.CpModel()

    # Scenario changes are applied to the actual candidate passes.
    available_passes = []

    for pid, sat, station, ps, pe, quality in PASSES:
        if scenario == "Ground Station Outage":
            if station == outage_station:
                continue

        if scenario == "Link Degradation":
            quality *= degradation

        rate = STATIONS[station] * quality

        # Round transmission duration UP to whole minutes.
        available_passes.append({
            "pass_id": pid,
            "satellite": sat,
            "station": station,
            "start": ps,
            "end": pe,
            "quality": quality,
            "rate": rate
        })

    candidates = []
    task_choices = {t[0]: [] for t in TASKS}
    sat_intervals = {s: [] for s in SATELLITES}
    station_intervals = {s: [] for s in STATIONS}
    storage_intervals = {s: [] for s in SATELLITES}
    power_terms = {s: [] for s in SATELLITES}

    for task in TASKS:
        tid, kind, priority, duration, data, deadline, power = task

        for sat, sd in SATELLITES.items():
            if kind not in sd["capabilities"]:
                continue

            for p in available_passes:
                if p["satellite"] != sat:
                    continue

                tx_duration = max(
                    1, math.ceil(data * 8 / (p["rate"] * 60))
                )

                if p["start"] + tx_duration > p["end"]:
                    continue

                if p["start"] + tx_duration > deadline:
                    continue

                i = len(candidates)
                present = model.NewBoolVar(f"choose_{i}")

                start = model.NewIntVar(
                    0, deadline - duration, f"start_{i}"
                )
                end = model.NewIntVar(
                    duration, deadline, f"end_{i}"
                )
                model.Add(end == start + duration)

                tx_start = model.NewIntVar(
                    p["start"], p["end"] - tx_duration,
                    f"tx_start_{i}"
                )
                tx_end = model.NewIntVar(
                    p["start"] + tx_duration, p["end"],
                    f"tx_end_{i}"
                )
                model.Add(tx_end == tx_start + tx_duration)

                # A task must finish before its data is transmitted.
                model.Add(end <= tx_start).OnlyEnforceIf(present)

                exec_interval = model.NewOptionalIntervalVar(
                    start, duration, end, present, f"exec_{i}"
                )
                tx_interval = model.NewOptionalIntervalVar(
                    tx_start, tx_duration, tx_end, present,
                    f"transmit_{i}"
                )

                sat_intervals[sat].extend(
                    [exec_interval, tx_interval]
                )
                station_intervals[p["station"]].append(tx_interval)
                power_terms[sat].append(present * power)
                task_choices[tid].append(present)

                # Task data occupies storage from execution end
                # until its downlink finishes.
                storage_duration = model.NewIntVar(
                    0, deadline, f"stored_duration_{i}"
                )
                model.Add(
                    storage_duration == tx_end - end
                ).OnlyEnforceIf(present)

                storage_interval = model.NewOptionalIntervalVar(
                    end, storage_duration, tx_end, present,
                    f"storage_{i}"
                )
                storage_intervals[sat].append(
                    (storage_interval, data)
                )

                candidates.append({
                    "index": i,
                    "task": task,
                    "satellite": sat,
                    "pass": p,
                    "present": present,
                    "start": start,
                    "end": end,
                    "tx_start": tx_start,
                    "tx_end": tx_end,
                    "tx_duration": tx_duration,
                    "exec_interval": exec_interval,
                    "tx_interval": tx_interval
                })

    # Each task can be selected at most once.
    for tid, choices in task_choices.items():
        if choices:
            model.Add(sum(choices) <= 1)

    # Satellite activity cannot overlap.
    for sat in SATELLITES:
        if sat_intervals[sat]:
            model.AddNoOverlap(sat_intervals[sat])

        model.Add(
            sum(power_terms[sat]) <= SATELLITES[sat]["power"]
        )

        # Conservative storage constraint: initial occupied storage
        # is reserved throughout the modeled horizon.
        items = storage_intervals[sat]
        if items:
            horizon = max(t[5] for t in TASKS)
            baseline = model.NewIntervalVar(
                0, horizon, horizon, f"initial_storage_{sat}"
            )
            all_intervals = [baseline]
            all_demands = [SATELLITES[sat]["initial"]]

            for interval, data in items:
                all_intervals.append(interval)
                all_demands.append(data)

            model.AddCumulative(
                all_intervals,
                all_demands,
                SATELLITES[sat]["storage"]
            )

    # A ground station can serve only one downlink at a time.
    for station in STATIONS:
        if station_intervals[station]:
            model.AddNoOverlap(station_intervals[station])

    # Maximize total priority of successfully scheduled tasks.
    model.Maximize(
        sum(
            c["present"] * (c["task"][2] * 100 + 1)
            for c in candidates
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15
    status = solver.Solve(model)

    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return status_name, pd.DataFrame(), 0, len(candidates)

    rows = []

    for c in candidates:
        if solver.Value(c["present"]) != 1:
            continue

        task = c["task"]
        p = c["pass"]

        rows.append({
            "Task": task[0],
            "Type": task[1],
            "Priority": task[2],
            "Satellite": c["satellite"],
            "Station": p["station"],
            "Pass": p["pass_id"],
            "Start": solver.Value(c["start"]),
            "End": solver.Value(c["end"]),
            "Downlink Start": solver.Value(c["tx_start"]),
            "Downlink End": solver.Value(c["tx_end"]),
            "Data MB": task[4],
            "Link Quality": round(p["quality"], 2)
        })

    result = pd.DataFrame(rows).sort_values("Start")
    utility = int(result["Priority"].sum()) if not result.empty else 0

    return status_name, result, utility, len(candidates)


# ---------------- INTERFACE ----------------

st.title("🛰️ Intelligent Satellite Mission Control")
st.caption(
    "Constraint-based scheduling prototype | Simulated orbital passes"
)

scenario = st.sidebar.selectbox(
    "Mission scenario",
    [
        "Normal Mission",
        "Urgent Tasking",
        "Ground Station Outage",
        "Link Degradation"
    ]
)

outage_station = "GS-01"
degradation = 0.5

if scenario == "Ground Station Outage":
    outage_station = st.sidebar.selectbox(
        "Station to disable", list(STATIONS.keys())
    )

if scenario == "Link Degradation":
    degradation = st.sidebar.slider(
        "Link quality multiplier",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1
    )

if scenario == "Urgent Tasking":
    st.info(
        "Urgent task priorities are not yet changed in this version. "
        "This selection currently uses normal task priorities."
    )

run = st.sidebar.button("Run optimizer", type="primary")

if "run_optimizer" not in st.session_state:
    st.session_state.run_optimizer = True

if run:
    st.session_state.run_optimizer = True

if st.session_state.run_optimizer:
    with st.spinner("Optimizing task and downlink schedule..."):
        status, result, utility, candidate_count = optimize_mission(
            scenario, outage_station, degradation
        )

    st.session_state.result = result
    st.session_state.status = status
    st.session_state.utility = utility
    st.session_state.candidate_count = candidate_count
    st.session_state.scenario = scenario
    st.session_state.run_optimizer = False

result = st.session_state.get("result", pd.DataFrame())
status = st.session_state.get("status", "Not run")
utility = st.session_state.get("utility", 0)
candidate_count = st.session_state.get("candidate_count", 0)

if st.session_state.get("scenario") != scenario:
    st.warning("Scenario changed. Click Run optimizer to calculate new results.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Scheduled Tasks", len(result), f"of {len(TASKS)}")
c2.metric("Unscheduled Tasks", len(TASKS) - len(result))
c3.metric("Mission Utility", utility)
c4.metric("Solver Status", status)

st.caption(f"Candidate combinations evaluated: {candidate_count}")

if scenario == "Ground Station Outage":
    st.warning(f"{outage_station} is excluded from all candidate downlinks.")
elif scenario == "Link Degradation":
    st.warning(
        f"Effective link quality is multiplied by {degradation:.1f}. "
        "Lower data rates may make passes infeasible."
    )

if not result.empty:
    st.subheader("Satellite Task Timeline")
    fig = px.timeline(
        result,
        x_start="Start",
        x_end="End",
        y="Satellite",
        color="Type",
        hover_data=["Task", "Priority", "Station"]
    )
    fig.update_layout(
        xaxis_title="Mission time (minutes)",
        yaxis_title="Satellite"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ground-Station Downlink Schedule")
    downlinks = result.rename(columns={
        "Downlink Start": "tx_start",
        "Downlink End": "tx_end"
    })

    fig2 = px.timeline(
        downlinks,
        x_start="tx_start",
        x_end="tx_end",
        y="Station",
        color="Satellite",
        hover_data=["Task", "Pass", "Link Quality"]
    )
    fig2.update_layout(
        xaxis_title="Mission time (minutes)",
        yaxis_title="Ground station"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Optimized Schedule")
    st.dataframe(result, use_container_width=True)

    skipped = sorted(set(t[0] for t in TASKS) - set(result["Task"]))
    st.write("Unscheduled tasks:", ", ".join(skipped) if skipped else "None")

else:
    st.info("No feasible schedule was returned. Try another scenario.")

st.caption(
    "Research prototype using synthetic task and pass data. "
    "Not connected to live satellite telemetry or real orbital ephemerides."
)

