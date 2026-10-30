import pandas as pd
import plotly.express as px
import streamlit as st
from .data_processing import add_observed_sla_breach, FEATURE_COLUMNS
from .prediction import load_model, load_metadata

def render_dashboard(df):
    st.subheader("Decision-Support Dashboard")
    work = add_observed_sla_breach(df)
    completed = work[work["Status"].eq("Completed")] if "Status" in work else work.iloc[0:0]
    pending = work[work["Status"].eq("Pending")] if "Status" in work else work.iloc[0:0]
    observed = completed.dropna(subset=["SLA_Breach"]) if "SLA_Breach" in completed else completed.iloc[0:0]
    breaches = int((observed["SLA_Breach"] == 1).sum()) if not observed.empty else 0
    rate = (breaches / len(observed) * 100) if len(observed) else 0.0

    a,b,c,d = st.columns(4)
    a.metric("Total requests", f"{len(work):,}")
    b.metric("Pending", f"{len(pending):,}")
    c.metric("Completed", f"{len(completed):,}")
    d.metric("Observed SLA breach rate", f"{rate:.1f}%", help="Calculated only from completed requests with known processing days.")

    st.caption("Decision support only: the model estimates risk; authorized staff must review context and make the final service decision.")
    if not pending.empty:
        model = load_model()
        meta = load_metadata()
        usable = pending.dropna(subset=meta["feature_columns"]).copy()
        if usable.empty:
            st.info("Pending requests do not contain enough feature values for a risk estimate.")
        else:
            usable["Estimated_SLA_Breach_Risk"] = model.predict_proba(usable[meta["feature_columns"]])[:, list(model.named_steps["classifier"].classes_).index(1)]
            usable["Risk_Band"] = pd.cut(usable["Estimated_SLA_Breach_Risk"], bins=[-0.001, 0.4, 0.7, 1.001], labels=["Lower", "Medium", "Higher"])
            st.markdown("#### Pending request review queue")
            display_cols = [c for c in ["Request_ID", "Service_Type", "Priority", "Assigned_Office", "Queue_Length", "SLA_Days", "Estimated_SLA_Breach_Risk", "Risk_Band"] if c in usable]
            st.dataframe(usable.sort_values("Estimated_SLA_Breach_Risk", ascending=False)[display_cols], use_container_width=True)
            st.markdown("#### Suggested human follow-up")
            st.write("- Review higher-risk requests first, especially urgent or time-sensitive cases.")
            st.write("- Confirm required documents and check whether the queue length reflects current workload.")
            st.write("- Contact the assigned office for context; do not automatically deny, cancel, or deprioritize a student's request.")
    if not observed.empty:
        observed = observed.copy()
        observed["SLA outcome"] = observed["SLA_Breach"].map({0: "Within SLA", 1: "SLA breach"})
        st.plotly_chart(px.bar(observed.groupby(["Assigned_Office", "SLA outcome"], dropna=False).size().reset_index(name="Requests"),
                               x="Assigned_Office", y="Requests", color="SLA outcome", barmode="group",
                               title="Observed SLA outcomes by office"), use_container_width=True)
    else:
        st.info("No completed requests with known processing times are available to calculate observed SLA outcomes.")
