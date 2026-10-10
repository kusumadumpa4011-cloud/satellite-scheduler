
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

STATIONS = {"GS-01": 20, "GS-02": 15, "GS-03": 10}

PASSES = [
    ("P01","SAT-01","GS-01",0,10,.95),
    ("P02","SAT-02","GS-02",5,15,.90),
    ("P03","SAT-03","GS-03",10,20,.85),
    ("P04","SAT-01","GS-02",15,25,.80),
    ("P05","SAT-02","GS-01",20,30,.95),
    ("P06","SAT-03","GS-02",25,35,.75),
    ("P07","SAT-01","GS-03",30,40,.90),
    ("P08","SAT-02","GS-03",35,45,.85),
    ("P09","SAT-03","GS-01",40,50,.95),
    ("P10","SAT-01","GS-01",45,55,.70),
    ("P11","SAT-02","GS-02",50,60,.90),
    ("P12","SAT-03","GS-03",55,65,.80)
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

def optimize_mission(scenario, outage_station, quality_factor):
    model = cp_model.CpModel()
    candidates = []
    task_choices = {t[0]: [] for t in TASKS}
    sat_intervals = {s: [] for s in SATELLITES}
    station_intervals = {s: [] for s in STATIONS}
    power_terms = {s: [] for s in SATELLITES}
    storage_items = {s: [] for s in SATELLITES}

    # Apply the selected scenario to actual visibility passes.
    usable_passes = []
    for pid, sat, station, ps, pe, quality in PASSES:
        if scenario == "Ground Station Outage" and station == outage_station:
            continue

        if scenario == "Link Degradation":
            quality *= quality_factor

        rate = STATIONS[station] * quality
        usable_passes.append({
            "id": pid, "sat": sat, "station": station,
            "start": ps, "end": pe, "quality": quality, "rate": rate
        })

    for task in TASKS:
        tid, kind, priority, duration, data, deadline, power = task

        for sat, sd in SATELLITES.items():
            if kind not in sd["capabilities"]:
                continue

            for p in usable_passes:
                if p["sat"] != sat:
                    continue

                tx_duration = max(
                    1, math.ceil(data * 8 / (p["rate"] * 60))
                )

                if p["start"] + tx_duration > p["end"]:
                    continue

                if p["start"] + tx_duration > deadline:
                    continue

                i = len(candidates)
                chosen = model.NewBoolVar(f"chosen_{i}")

                start = model.NewIntVar(0, deadline - duration, f"start_{i}")
                end = model.NewIntVar(duration, deadline, f"end_{i}")
                model.Add(end == start + duration)

                tx_start = model.NewIntVar(
                    p["start"], p["end"] - tx_duration, f"txstart_{i}"
                )
                tx_end = model.NewIntVar(
                    p["start"] + tx_duration, p["end"], f"txend_{i}"
                )
                model.Add(tx_end == tx_start + tx_duration)
                model.Add(end <= tx_start).OnlyEnforceIf(chosen)

                exec_interval = model.NewOptionalIntervalVar(
                    start, duration, end, chosen, f"exec_{i}"
                )
                tx_interval = model.NewOptionalIntervalVar(
                    tx_start, tx_duration, tx_end, chosen, f"tx_{i}"
                )

                sat_intervals[sat] += [exec_interval, tx_interval]
                station_intervals[p["station"]].append(tx_interval)
                power_terms[sat].append(chosen * power)
                task_choices[tid].append(chosen)

                # Storage is occupied between task completion and downlink.
                stored_duration = model.NewIntVar(
                    0, deadline, f"stored_duration_{i}"
                )
                model.Add(stored_duration == tx_end - end).OnlyEnforceIf(chosen)

                stored_interval = model.NewOptionalIntervalVar(
                    end, stored_duration, tx_end, chosen, f"storage_{i}"
                )
                storage_items[sat].append((stored_interval, data))

                candidates.append({
                    "chosen": chosen, "task": task, "sat": sat,
                    "pass": p, "start": start, "end": end,
                    "tx_start": tx_start, "tx_end": tx_end,
                    "tx_duration": tx_duration
                })

    for choices in task_choices.values():
        if choices:
            model.Add(sum(choices) <= 1)

    for sat, sd in SATELLITES.items():
        if sat_intervals[sat]:
            model.AddNoOverlap(sat_intervals[sat])

        if power_terms[sat]:
            model.Add(sum(power_terms[sat]) <= sd["power"])

        # Reserve initial occupied storage for the entire planning horizon.
        baseline = model.NewIntervalVar(0, 65, 65, f"initial_{sat}")
        intervals = [baseline]
        demands = [sd["initial"]]

        for interval, data in storage_items[sat]:
            intervals.append(interval)
            demands.append(data)

        model.AddCumulative(intervals, demands, sd["storage"])

    for station, intervals in station_intervals.items():
        if intervals:
            model.AddNoOverlap(intervals)

    model.Maximize(sum(
        c["chosen"] * (c["task"][2] * 100 + 1)
        for c in candidates
    ))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.StatusName(status), pd.DataFrame(), 0, len(candidates)

    rows = []
    for c in candidates:
        if solver.Value(c["chosen"]) == 0:
            continue

        t, p = c["task"], c["pass"]
        rows.append({
            "Task": t[0], "Type": t[1], "Priority": t[2],
            "Satellite": c["sat"], "Station": p["station"],
            "Pass": p["id"],
            "Start": solver.Value(c["start"]),
            "End": solver.Value(c["end"]),
            "Downlink Start": solver.Value(c["tx_start"]),
            "Downlink End": solver.Value(c["tx_end"]),
            "Data MB": t[4], "Link Quality": round(p["quality"], 2)
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("Start").reset_index(drop=True)

    utility = int(result["Priority"].sum()) if not result.empty else 0
    return solver.StatusName(status), result, utility, len(candidates)


# ---------------- STREAMLIT INTERFACE ----------------

st.title("🛰️ Intelligent Satellite Mission Control")
st.caption("Constraint-based scheduling prototype using simulated orbital passes")

scenario = st.sidebar.selectbox(
    "Mission scenario",
    ["Normal Mission", "Ground Station Outage", "Link Degradation"]
)

outage_station = "GS-01"
quality_factor = 0.5

if scenario == "Ground Station Outage":
    outage_station = st.sidebar.selectbox(
        "Unavailable station", list(STATIONS.keys())
    )

if scenario == "Link Degradation":
    quality_factor = st.sidebar.slider(
        "Link quality multiplier", 0.1, 0.9, 0.5, 0.1
    )

if st.sidebar.button("Run optimizer", type="primary"):
    with st.spinner("Solving the scheduling problem..."):
        status, result, utility, candidate_count = optimize_mission(
            scenario, outage_station, quality_factor
        )

    st.session_state["result"] = result
    st.session_state["status"] = status
    st.session_state["utility"] = utility
    st.session_state["candidate_count"] = candidate_count
    st.session_state["scenario"] = scenario
    st.session_state["station"] = outage_station
    st.session_state["factor"] = quality_factor

if "result" not in st.session_state:
    with st.spinner("Running initial mission optimization..."):
        status, result, utility, candidate_count = optimize_mission(
            "Normal Mission", outage_station, quality_factor
        )
    st.session_state.update({
        "result": result, "status": status, "utility": utility,
        "candidate_count": candidate_count,
        "scenario": "Normal Mission",
        "station": outage_station, "factor": quality_factor
    })

result = st.session_state["result"]
status = st.session_state["status"]
utility = st.session_state["utility"]

if st.session_state["scenario"] != scenario:
    st.warning("Scenario changed. Click Run optimizer to calculate the new schedule.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Scheduled Tasks", len(result), f"of {len(TASKS)}")
c2.metric("Unscheduled Tasks", len(TASKS) - len(result))
c3.metric("Mission Utility", utility)
c4.metric("Solver Status", status)

st.caption(
    f"Candidate combinations evaluated: {st.session_state['candidate_count']}"
)

if scenario == "Ground Station Outage":
    st.warning(f"{outage_station} is removed from the candidate visibility passes.")
elif scenario == "Link Degradation":
    st.warning(
        f"Effective link quality is multiplied by {quality_factor:.1f}. "
        "Reduced rates can make some downlinks infeasible."
    )

if result.empty:
    st.info("No feasible tasks were selected for this scenario.")
else:
    st.subheader("Satellite Task Timeline")

    # Numeric bar chart: avoids Plotly interpreting minutes as timestamps.
    task_plot = result.copy()
    task_plot["Duration"] = task_plot["End"] - task_plot["Start"]

    fig = px.bar(
        task_plot,
        x="Duration",
        y="Satellite",
        base="Start",
        color="Type",
        orientation="h",
        hover_data=["Task", "Priority", "Station", "Start", "End"]
    )
    fig.update_layout(
        xaxis_title="Mission time (minutes)",
        yaxis_title="Satellite",
        xaxis_type="linear"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ground-Station Downlink Schedule")
    downlink_plot = result.copy()
    downlink_plot["Transmission Duration"] = (
        downlink_plot["Downlink End"] - downlink_plot["Downlink Start"]
    )

    fig2 = px.bar(
        downlink_plot,
        x="Transmission Duration",
        y="Station",
        base="Downlink Start",
        color="Satellite",
        orientation="h",
        hover_data=["Task", "Pass", "Link Quality"]
    )
    fig2.update_layout(
        xaxis_title="Mission time (minutes)",
        yaxis_title="Ground station",
        xaxis_type="linear"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Optimized Schedule")
    st.dataframe(result, use_container_width=True)

    skipped = sorted(set(t[0] for t in TASKS) - set(result["Task"]))
    st.write("Unscheduled tasks:", ", ".join(skipped) if skipped else "None")

    # Basic consistency checks on the displayed schedule.
    st.subheader("Schedule Validation")
    checks = []

    checks.append((
        "No downlinks use failed station",
        scenario != "Ground Station Outage"
        or not (result["Station"] == outage_station).any()
    ))

    checks.append((
        "No downlink exceeds its visibility pass",
        all(
            row["Downlink Start"] >= next(
                p[3] for p in PASSES if p[0] == row["Pass"]
            )
            and row["Downlink End"] <= next(
                p[4] for p in PASSES if p[0] == row["Pass"]
            )
            for _, row in result.iterrows()
        )
    ))

    checks.append((
        "Task completes before downlink",
        (result["End"] <= result["Downlink Start"]).all()
    ))

    checks.append((
        "No station transmission overlaps",
        all(
            all(
                a["Downlink End"] <= b["Downlink Start"]
                or b["Downlink End"] <= a["Downlink Start"]
                for i, a in enumerate(group.to_dict("records"))
                for b in group.to_dict("records")[i+1:]
            )
            for _, group in result.groupby("Station")
        )
    ))

    for label, passed in checks:
        st.write(("✅ " if passed else "❌ ") + label)

st.caption(
    "Synthetic research prototype, not connected to live satellite telemetry. "
    "Power and storage use simplified planning constraints."
)
