import json
from pathlib import Path

import streamlit as st


DEMO_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "seed_data"
    / "demo_script.json"
)

with DEMO_SCRIPT_PATH.open(encoding="utf-8") as file:
    demo = json.load(file)

st.set_page_config(
    page_title="Pantry Agent Demo",
    page_icon="🥬",
    layout="wide",
)

st.title("Pantry Agent")
st.caption("Three-round scripted preview. Live agent events are not connected yet.")

with st.sidebar:
    st.header("Demo controls")

    round_number = st.selectbox(
        "Round",
        options=[item["round_number"] for item in demo["rounds"]],
    )

    auto_run = st.toggle(
        "Auto-run (preview only)",
        value=demo["settings"]["auto_run"],
    )

    auto_accept = st.toggle(
        "Auto-accept non-risk results (preview only)",
        value=demo["settings"]["auto_accept_non_risk"],
    )

round_data = next(
    item
    for item in demo["rounds"]
    if item["round_number"] == round_number
)

trigger = round_data["trigger"]

st.header(
    f"Demo Day {round_data['day_index'] + 1} · "
    f"Round {round_data['round_number']}"
)
st.caption(f"Virtual time: {round_data['at']}")

day_column, round_column, item_column, expiry_column = st.columns(4)

day_column.metric("Demo day", round_data["day_index"] + 1)
round_column.metric("Round", round_data["round_number"])
item_column.metric("Trigger item", trigger["item_name"])
expiry_column.metric(
    "Days until expiry",
    trigger["days_until_expiry"],
)

inventory_tab, preview_tab, timeline_tab = st.tabs(
    ["Opening inventory", "Script preview", "Live timeline"]
)

with inventory_tab:
    st.dataframe(
        round_data["opening_inventory"],
        hide_index=True,
        use_container_width=True,
    )

with preview_tab:
    st.info(
        "These are expected demo steps. They are not recorded agent actions."
    )

    for step_number, step in enumerate(
        round_data["expected_steps"],
        start=1,
    ):
        st.write(f"{step_number}. {step}")

with timeline_tab:
    st.info(
        "Actual Run.steps will appear here after backend integration."
    )

if "required_memory_assertion" in round_data:
    memory = round_data["required_memory_assertion"]
    st.info(
        "Round 2 memory check: "
        f"{memory['retrieved_trace_alias']}"
    )

if "constraint_change" in round_data:
    constraint = round_data["constraint_change"]
    allergies = ", ".join(constraint["allergies"])

    st.warning(
        "Human confirmation required. "
        f"Guest allergies: {allergies}."
    )