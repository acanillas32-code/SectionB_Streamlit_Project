import pandas as pd
import plotly.express as px
import streamlit as st
from .data_processing import add_observed_sla_breach

def render_eda(df):
    st.subheader("Exploratory Data Analysis")
    work = add_observed_sla_breach(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(work):,}")
    c2.metric("Columns", len(work.columns))
    c3.metric("Duplicate request IDs", int(work["Request_ID"].duplicated().sum()) if "Request_ID" in work else 0)

    st.markdown("#### Descriptive statistics")
    st.dataframe(work.describe(include="all").transpose(), use_container_width=True)

    st.markdown("#### Interactive charts")
    categorical_options = [c for c in ["Service_Type", "Status", "Priority", "Requested_Channel", "Assigned_Office", "Requester_Type"] if c in work.columns]
    if categorical_options:
        selected = st.selectbox("Choose a category for the count chart", categorical_options)
        counts = work[selected].fillna("Missing").astype(str).value_counts().rename_axis(selected).reset_index(name="Count")
        st.plotly_chart(px.bar(counts, x=selected, y="Count", title=f"Requests by {selected}", color=selected), use_container_width=True)

    if {"Queue_Length", "SLA_Days"}.issubset(work.columns):
        st.plotly_chart(px.scatter(work, x="Queue_Length", y="SLA_Days", color="Status" if "Status" in work else None,
                                   hover_data=[c for c in ["Request_ID", "Service_Type", "Priority"] if c in work],
                                   title="Queue length versus SLA days"), use_container_width=True)

    if "Processing_Days" in work.columns:
        valid = work.dropna(subset=["Processing_Days"])
        if not valid.empty:
            st.plotly_chart(px.histogram(valid, x="Processing_Days", color="Status" if "Status" in valid else None,
                                         nbins=15, title="Observed processing days (known values only)"), use_container_width=True)

    if "SLA_Breach" in work.columns:
        observed = work.dropna(subset=["SLA_Breach"])
        if not observed.empty:
            observed["SLA_Breach_Label"] = observed["SLA_Breach"].map({0: "Within SLA", 1: "SLA breach"})
            st.plotly_chart(px.pie(observed, names="SLA_Breach_Label", title="Observed SLA outcomes — completed requests only"), use_container_width=True)
