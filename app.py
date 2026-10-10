
import math
import time
from collections import defaultdict

import pandas as pd
import plotly.express as px
import streamlit as st
from ortools.sat.python import cp_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Satellite Scheduler",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ Intelligent Satellite Communication & Task Scheduling")
st.caption(
    "CP-SAT constraint optimization vs. priority-based greedy scheduling. "
    "All data are simulated, not live satellite telemetry."
)


# ============================================================
# SIMULATED INPUT DATA
# Time is represented as minutes from the beginning of the mission.
# Data sizes are MB; rates are MB/minute.
# ============================================================

def create_data():
    satellites = pd.DataFrame([
        {"satellite": "SAT-1"},
        {"satellite": "SAT-2"},
        {"satellite": "SAT-3"},
    ])

    stations = pd.DataFrame([
        {"station": "GS-A"},
        {"station": "GS-B"},
        {"station": "GS-C"},
    ])

    passes = pd.DataFrame([
        {"pass_id": "P01", "satellite": "SAT-1", "station": "GS-A",
         "start": 0, "end": 20, "rate": 12},
        {"pass_id": "P02", "satellite": "SAT-2", "station": "GS-B",
         "start": 5, "end": 25, "rate": 10},
        {"pass_id": "P03", "satellite": "SAT-3", "station": "GS-C",
         "start": 10, "end": 30, "rate": 11},
        {"pass_id": "P04", "satellite": "SAT-1", "station": "GS-B",
         "start": 25, "end": 45, "rate": 12},
        {"pass_id": "P05", "satellite": "SAT-2", "station": "GS-C",
         "start": 30, "end": 50, "rate": 10},
        {"pass_id": "P06", "satellite": "SAT-3", "station": "GS-A",
         "start": 35, "end": 55, "rate": 11},
        {"pass_id": "P07", "satellite": "SAT-1", "station": "GS-C",
         "start": 55, "end": 75, "rate": 12},
        {"pass_id": "P08", "satellite": "SAT-2", "station": "GS-A",
         "start": 60, "end": 80, "rate": 10},
        {"pass_id": "P09", "satellite": "SAT-3", "station": "GS-B",
         "start": 65, "end": 85, "rate": 11},
        {"pass_id": "P10", "satellite": "SAT-1", "station": "GS-A",
         "start": 90, "end": 110, "rate": 12},
        {"pass_id": "P11", "satellite": "SAT-2", "station": "GS-B",
         "start": 95, "end": 115, "rate": 10},
        {"pass_id": "P12", "satellite": "SAT-3", "station": "GS-C",
         "start": 100, "end": 120, "rate": 11},
    ])

    task_rows = [
        ("T01", "SAT-1", 10, 35, 100),
        ("T02", "SAT-2", 8, 45, 90),
        ("T03", "SAT-3", 9, 40, 100),
        ("T04", "SAT-1", 7, 65, 120),
        ("T05", "SAT-2", 6, 70, 100),
        ("T06", "SAT-3", 8, 70, 110),
        ("T07", "SAT-1", 5, 90, 80),
        ("T08", "SAT-2", 9, 90, 120),
        ("T09", "SAT-3", 7, 95, 90),
        ("T10", "SAT-1", 8, 115, 100),
        ("T11", "SAT-2", 5, 120, 80),
        ("T12", "SAT-3", 6, 120, 90),
        ("T13", "SAT-1", 4, 80, 70),
        ("T14", "SAT-2", 7, 105, 90),
        ("T15", "SAT-3", 5, 110, 80),
        ("T16", "SAT-1", 9, 50, 110),
        ("T17", "SAT-2", 4, 85, 70),
        ("T18", "SAT-3", 8, 75, 100),
    ]

    tasks = pd.DataFrame(
        task_rows,
        columns=["task_id", "satellite", "priority", "deadline", "data_mb"],
    )

    return satellites, stations, passes, tasks


# ============================================================
# SCENARIO CONFIGURATION
# Multiple Events applies BOTH disruptions to the same inputs:
#   1. One ground station is unavailable.
#   2. Link rate is reduced on all remaining eligible passes.
# ============================================================

SCENARIOS = [
    "Normal Mission",
    "Urgent Tasking",
    "Ground Station Outage",
    "Link Degradation",
    "Multiple Events",
]


def apply_scenario(passes, tasks, scenario, outage_station, rate_factor):
    adjusted_passes = passes.copy()
    adjusted_tasks = tasks.copy()

    unavailable_stations = set()
    effective_rate_factor = 1.0

    if scenario == "Urgent Tasking":
        # Give one task higher importance and an earlier deadline.
        mask = adjusted_tasks["task_id"] == "T07"
        adjusted_tasks.loc[mask, "priority"] = 15
        adjusted_tasks.loc[mask, "deadline"] = 75

    elif scenario == "Ground Station Outage":
        unavailable_stations.add(outage_station)

    elif scenario == "Link Degradation":
        effective_rate_factor = rate_factor

    elif scenario == "Multiple Events":
        # IMPORTANT: both disruptions are active simultaneously.
        unavailable_stations.add(outage_station)
        effective_rate_factor = rate_factor

    adjusted_passes["effective_rate"] = (
        adjusted_passes["rate"] * effective_rate_factor
    )

    # A pass at an unavailable station cannot be used.
    adjusted_passes["available"] = ~adjusted_passes["station"].isin(
        unavailable_stations
    )

    return adjusted_passes, adjusted_tasks, unavailable_stations


# ============================================================
# CANDIDATE GENERATION
# Each candidate is one compatible task/pass pair.
# Transmission duration is rounded up to whole minutes.
# ============================================================

def generate_candidates(passes, tasks):
    candidates = []

    for task in tasks.itertuples(index=False):
        for p in passes.itertuples(index=False):
            if not p.available:
                continue

            if task.satellite != p.satellite:
                continue

            if p.effective_rate <= 0:
                continue

            duration = max(
                1,
                math.ceil(task.data_mb / p.effective_rate),
            )

            latest_end = min(int(p.end), int(task.deadline))

            if int(p.start) + duration > latest_end:
                continue

            candidates.append({
                "task_id": task.task_id,
                "satellite": task.satellite,
                "priority": int(task.priority),
                "deadline": int(task.deadline),
                "data_mb": int(task.data_mb),
                "pass_id": p.pass_id,
                "station": p.station,
                "pass_start": int(p.start),
                "pass_end": int(p.end),
                "rate": float(p.effective_rate),
                "duration": int(duration),
                "latest_start": latest_end - duration,
            })

    return candidates


# ============================================================
# CP-SAT OPTIMIZER
# Every candidate has an optional interval.
# A task can be scheduled at most once.
# Stations and satellites cannot have overlapping transmissions.
# ============================================================

def optimize_cpsat(candidates, tasks, time_limit=10):
    if not candidates:
        return pd.DataFrame(), "NO_CANDIDATES", 0.0

    model = cp_model.CpModel()

    selected = {}
    intervals = {}
    starts = {}

    by_task = defaultdict(list)
    by_station = defaultdict(list)
    by_satellite = defaultdict(list)

    for i, c in enumerate(candidates):
        key = i

        selected[key] = model.NewBoolVar(f"selected_{i}")

        starts[key] = model.NewIntVar(
            c["pass_start"],
            c["latest_start"],
            f"start_{i}",
        )

        intervals[key] = model.NewOptionalIntervalVar(
            starts[key],
            c["duration"],
            starts[key] + c["duration"],
            selected[key],
            f"interval_{i}",
        )

        by_task[c["task_id"]].append(selected[key])
        by_station[c["station"]].append(intervals[key])
        by_satellite[c["satellite"]].append(intervals[key])

    # A task can be assigned to at most one pass.
    for task_id, variables in by_task.items():
        model.Add(sum(variables) <= 1)

    # No overlapping transmissions on the same station or satellite.
    for station, station_intervals in by_station.items():
        model.AddNoOverlap(station_intervals)

    for satellite, satellite_intervals in by_satellite.items():
        model.AddNoOverlap(satellite_intervals)

    # Maximize the total selected task priority.
    priority_by_task = {
        row.task_id: int(row.priority)
        for row in tasks.itertuples(index=False)
    }

    model.Maximize(
        sum(
            priority_by_task[c["task_id"]] * selected[i]
            for i, c in enumerate(candidates)
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8

    start_time = time.perf_counter()
    status_code = solver.Solve(model)
    runtime = time.perf_counter() - start_time

    status = solver.StatusName(status_code)

    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return pd.DataFrame(), status, runtime

    result = []

    for i, c in enumerate(candidates):
        if solver.Value(selected[i]):
            row = c.copy()
            row["start"] = int(solver.Value(starts[i]))
            row["end"] = row["start"] + row["duration"]
            row["status"] = "Scheduled"
            result.append(row)

    return pd.DataFrame(result), status, runtime


# ============================================================
# GREEDY BASELINE
# Process higher-priority tasks first, then earlier deadlines.
# For each task, find the earliest non-conflicting feasible slot.
# ============================================================

def run_greedy(candidates, tasks):
    if not candidates:
        return pd.DataFrame()

    task_order = tasks.sort_values(
        ["priority", "deadline"],
        ascending=[False, True],
    )

    selected_rows = []
    used_tasks = set()

    station_intervals = defaultdict(list)
    satellite_intervals = defaultdict(list)

    def conflicts(existing, start, end):
        return any(
            start < old_end and end > old_start
            for old_start, old_end in existing
        )

    for task in task_order.itertuples(index=False):
        options = [
            c for c in candidates
            if c["task_id"] == task.task_id
        ]

        options.sort(key=lambda c: (c["pass_start"], c["duration"]))

        for c in options:
            if c["task_id"] in used_tasks:
                continue

            # Search for the earliest valid minute in this pass.
            assigned = False

            for start in range(
                c["pass_start"],
                c["latest_start"] + 1,
            ):
                end = start + c["duration"]

                if conflicts(station_intervals[c["station"]], start, end):
                    continue

                if conflicts(
                    satellite_intervals[c["satellite"]], start, end
                ):
                    continue

                row = c.copy()
                row["start"] = start
                row["end"] = end
                row["status"] = "Scheduled"

                selected_rows.append(row)
                used_tasks.add(c["task_id"])

                station_intervals[c["station"]].append((start, end))
                satellite_intervals[c["satellite"]].append((start, end))

                assigned = True
                break

            if assigned:
                break

    return pd.DataFrame(selected_rows)


# ============================================================
# VALIDATION
# Check deadline, pass window, station outage, rate duration,
# task uniqueness, and resource overlaps.
# ============================================================

def validate_schedule(schedule, passes, tasks, unavailable_stations):
    issues = []

    if schedule.empty:
        return ["No tasks scheduled."]

    if schedule["task_id"].duplicated().any():
        issues.append("A task was scheduled more than once.")

    task_lookup = tasks.set_index("task_id").to_dict("index")
    pass_lookup = passes.set_index("pass_id").to_dict("index")

    for row in schedule.itertuples(index=False):
        p = pass_lookup[row.pass_id]
        task = task_lookup[row.task_id]

        if row.station in unavailable_stations:
            issues.append(
                f"{row.task_id}: assigned to unavailable station {row.station}."
            )

        if row.start < p["start"] or row.end > p["end"]:
            issues.append(
                f"{row.task_id}: transmission lies outside its pass window."
            )

        if row.end > task["deadline"]:
            issues.append(f"{row.task_id}: deadline violated.")

        if row.duration != max(
            1, math.ceil(task["data_mb"] / p["effective_rate"])
        ):
            issues.append(
                f"{row.task_id}: duration does not match the modeled rate."
            )

    for resource in ["station", "satellite"]:
        for resource_name, group in schedule.groupby(resource):
            group = group.sort_values("start")
            previous_end = None

            for row in group.itertuples(index=False):
                if previous_end is not None and row.start < previous_end:
                    issues.append(
                        f"{resource} {resource_name}: overlapping transmissions."
                    )
                    break
                previous_end = max(
                    previous_end or row.end,
                    row.end,
                )

    return issues


# ============================================================
# METRICS
# ============================================================

def summarize(schedule, tasks, runtime=None, solver_status=None):
    if schedule.empty:
        return {
            "Tasks scheduled": 0,
            "Priority utility": 0,
            "Unscheduled tasks": len(tasks),
            "Runtime (seconds)": runtime,
            "Solver status": solver_status or "Not applicable",
        }

    return {
        "Tasks scheduled": int(schedule["task_id"].nunique()),
        "Priority utility": int(schedule["priority"].sum()),
        "Unscheduled tasks": int(
            len(tasks) - schedule["task_id"].nunique()
        ),
        "Runtime (seconds)": (
            round(runtime, 4) if runtime is not None else None
        ),
        "Solver status": solver_status or "Not applicable",
    }


# ============================================================
# STREAMLIT CONTROLS
# ============================================================

with st.sidebar:
    st.header("Scenario controls")

    scenario = st.selectbox(
        "Select stress-test scenario",
        SCENARIOS,
        index=0,
    )

    outage_station = st.selectbox(
        "Station affected by outage",
        ["GS-A", "GS-B", "GS-C"],
        index=2,
        help="Used by Ground Station Outage and Multiple Events.",
    )

    rate_factor = st.slider(
        "Remaining link-rate multiplier",
        min_value=0.20,
        max_value=1.00,
        value=0.55,
        step=0.05,
        help="0.55 means 55% of the original rate, or a 45% reduction.",
    )

    time_limit = st.slider(
        "CP-SAT time limit (seconds)",
        min_value=1,
        max_value=30,
        value=10,
    )

    st.info(
        "Multiple Events activates both the selected station outage "
        "and reduced link rate in one run."
    )


# ============================================================
# RUN THE APPLICATION
# ============================================================

satellites, stations, base_passes, base_tasks = create_data()

passes, tasks, unavailable_stations = apply_scenario(
    base_passes,
    base_tasks,
    scenario,
    outage_station,
    rate_factor,
)

candidates = generate_candidates(passes, tasks)

st.subheader("1. Scenario configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Satellites", len(satellites))

with col2:
    st.metric("Ground stations", len(stations))

with col3:
    st.metric("Candidate assignments", len(candidates))

st.write(f"**Selected scenario:** {scenario}")

if scenario in ("Ground Station Outage", "Multiple Events"):
    st.warning(
        f"Station outage active: {', '.join(sorted(unavailable_stations))}"
    )

if scenario in ("Link Degradation", "Multiple Events"):
    st.warning(
        f"Link degradation active: {rate_factor:.0%} of original data rate."
    )

st.subheader("2. Input data")

with st.expander("View tasks"):
    st.dataframe(tasks, use_container_width=True)

with st.expander("View communication passes"):
    st.dataframe(passes, use_container_width=True)

st.subheader("3. Run schedulers")

if st.button("🚀 Run Greedy and CP-SAT benchmark", type="primary"):
    with st.spinner("Solving scheduling problem..."):
        greedy_start = time.perf_counter()
        greedy_schedule = run_greedy(candidates, tasks)
        greedy_runtime = time.perf_counter() - greedy_start

        cpsat_schedule, solver_status, cpsat_runtime = optimize_cpsat(
            candidates,
            tasks,
            time_limit=time_limit,
        )

        st.session_state["run_results"] = {
            "scenario": scenario,
            "passes": passes,
            "tasks": tasks,
            "unavailable_stations": unavailable_stations,
            "greedy_schedule": greedy_schedule,
            "cpsat_schedule": cpsat_schedule,
            "greedy_runtime": greedy_runtime,
            "cpsat_runtime": cpsat_runtime,
            "solver_status": solver_status,
        }


# Keep the latest run visible after Streamlit reruns.
if "run_results" not in st.session_state:
    st.info("Select a scenario, then click **Run Greedy and CP-SAT benchmark**.")
    st.stop()

results = st.session_state["run_results"]

if results["scenario"] != scenario:
    st.warning(
        "The scenario controls changed since the last run. "
        "Click the benchmark button again to calculate updated results."
    )
    st.stop()

greedy_schedule = results["greedy_schedule"]
cpsat_schedule = results["cpsat_schedule"]
tasks = results["tasks"]
passes = results["passes"]
unavailable_stations = results["unavailable_stations"]

st.subheader("4. Benchmark results")

greedy_metrics = summarize(
    greedy_schedule,
    tasks,
    runtime=results["greedy_runtime"],
    solver_status="Greedy heuristic",
)

cpsat_metrics = summarize(
    cpsat_schedule,
    tasks,
    runtime=results["cpsat_runtime"],
    solver_status=results["solver_status"],
)

comparison = pd.DataFrame(
    [
        {"Method": "Greedy", **greedy_metrics},
        {"Method": "CP-SAT", **cpsat_metrics},
    ]
).set_index("Method")

st.dataframe(comparison, use_container_width=True)

base_utility = greedy_metrics["Priority utility"]
optimized_utility = cpsat_metrics["Priority utility"]

if base_utility > 0:
    improvement = (
        (optimized_utility - base_utility) / base_utility
    ) * 100
    st.metric(
        "CP-SAT priority-utility change vs. Greedy",
        f"{improvement:+.1f}%",
    )
else:
    st.caption(
        "Percentage change is unavailable because Greedy utility is zero."
    )

st.caption(
    "Priority utility is the sum of the selected tasks' priority values. "
    "It is a modeled score, not real-world satellite throughput."
)

st.subheader("5. Schedule validation")

for method_name, schedule in [
    ("Greedy", greedy_schedule),
    ("CP-SAT", cpsat_schedule),
]:
    issues = validate_schedule(
        schedule,
        passes,
        tasks,
        unavailable_stations,
    )

    with st.expander(f"{method_name} validation"):
        if not issues:
            st.success("All implemented validation checks passed.")
        else:
            st.error(f"{len(issues)} validation issue(s) found.")
            for issue in issues:
                st.write(f"- {issue}")

st.subheader("6. Optimized schedule")

if cpsat_schedule.empty:
    st.warning(
        f"CP-SAT returned no schedule. Solver status: "
        f"{results['solver_status']}"
    )
else:
    display_columns = [
        "task_id",
        "satellite",
        "station",
        "pass_id",
        "start",
        "end",
        "duration",
        "priority",
        "deadline",
        "data_mb",
        "rate",
    ]

    st.dataframe(
        cpsat_schedule[display_columns].sort_values("start"),
        use_container_width=True,
    )

    chart_df = cpsat_schedule.copy()
    chart_df["Task"] = (
        chart_df["task_id"] + " | " + chart_df["station"]
    )

    fig = px.timeline(
        chart_df,
        x_start="start",
        x_end="end",
        y="station",
        color="satellite",
        hover_data=[
            "task_id",
            "pass_id",
            "priority",
            "deadline",
            "duration",
            "rate",
        ],
        title="CP-SAT transmission schedule",
    )

    fig.update_yaxes(autorange="reversed", title="Ground station")
    fig.update_xaxes(title="Mission time (minutes)")
    fig.update_layout(legend_title="Satellite")

    st.plotly_chart(fig, use_container_width=True)

st.subheader("7. Download results")

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        "Download CP-SAT schedule CSV",
        data=cpsat_schedule.to_csv(index=False).encode("utf-8"),
        file_name=f"cpsat_schedule_{scenario.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

with col2:
    st.download_button(
        "Download Greedy schedule CSV",
        data=greedy_schedule.to_csv(index=False).encode("utf-8"),
        file_name=f"greedy_schedule_{scenario.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Prototype limitation: the data, visibility windows, and communication "
    "rates are simulated. This application does not control or communicate "
    "with real satellites."
)
