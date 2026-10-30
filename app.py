from pathlib import Path
import pandas as pd
import streamlit as st
from modules.data_processing import validate_dataset
from modules.exploratory_analysis import render_eda
from modules.prediction import load_metadata, predict_one
from modules.dashboard import render_dashboard

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "student_service_requests.csv"

st.set_page_config(page_title="Section B | Student Service Request Monitoring", page_icon="📊", layout="wide")
st.title("📊 Student Service Request Monitoring")
st.caption("Section B · Data Mining Deployment · Systems Analysis and Design")

with st.sidebar:
    st.header("Navigation")
    page = st.selectbox("Choose a module", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
    st.divider()
    st.caption("Model status: PROVISIONAL")
    st.caption("Version: provisional-1.0")

@st.cache_data
def load_default_data():
    return pd.read_csv(DEFAULT_DATA)

if "active_df" not in st.session_state:
    st.session_state.active_df = load_default_data()
df = st.session_state.active_df.copy()

if page == "Home":
    st.subheader("Home")
    st.write("This app helps student-services staff inspect request data, explore workload patterns, estimate possible SLA-breach risk, and prioritize human review.")
    st.warning("Important provenance note: the uploaded midterm notebook contains exploratory analysis code but no trained model, and its data-generation code references an undefined variable `n`. The bundled model is therefore a clearly labelled PROVISIONAL model, not a verified reuse of the original midterm model.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Dataset rows", f"{len(df):,}")
    c2.metric("Dataset columns", len(df.columns))
    c3.metric("Pending requests", int(df["Status"].eq("Pending").sum()) if "Status" in df else 0)
    st.markdown("#### Modules")
    st.markdown("- **Dataset:** upload CSV, validate schema, inspect nulls and duplicates.\n- **Data Analysis:** statistics and interactive charts.\n- **Prediction:** enter request details and estimate SLA-breach risk.\n- **Dashboard:** KPIs and a review queue for pending requests.\n- **System Design:** requirements, actors, use-case diagram, and workflow.")
    st.markdown("#### How to use")
    st.write("1. Open Dataset to upload or validate the CSV. 2. Explore Data Analysis. 3. Use Prediction for a single hypothetical request. 4. Review the Dashboard. 5. Read System Design before deployment.")
    st.info("Use synthetic or properly authorized data for public deployment. Predictions are decision-support only and must not automatically deny student services.")

elif page == "Dataset":
    st.subheader("Dataset Upload and Validation")
    uploaded = st.file_uploader("Upload a student service requests CSV", type=["csv"])
    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            issues = validate_dataset(incoming)
            if issues:
                st.error("The uploaded file has validation issues:")
                for issue in issues:
                    st.write(f"- {issue}")
                st.caption("The current active dataset remains unchanged until the file passes validation.")
            else:
                st.session_state.active_df = incoming
                df = incoming.copy()
                st.success("CSV passed the required-column and basic value checks. It is now the active dataset.")
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    st.markdown("#### Active dataset")
    issues = validate_dataset(df)
    if issues:
        st.error("Active dataset validation:")
        for issue in issues:
            st.write(f"- {issue}")
    else:
        st.success("Required schema found.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Duplicate Request_ID", int(df["Request_ID"].duplicated().sum()) if "Request_ID" in df else 0)
    st.dataframe(df.head(30), use_container_width=True)
    st.markdown("#### Missing values")
    missing = df.isna().sum().rename("Missing values").to_frame()
    missing["Missing %"] = (missing["Missing values"] / max(len(df),1) * 100).round(1)
    st.dataframe(missing, use_container_width=True)
    st.download_button("Download active dataset as CSV", df.to_csv(index=False).encode("utf-8"), "student_service_requests_active.csv", "text/csv")

elif page == "Data Analysis":
    render_eda(df)

elif page == "Prediction":
    st.subheader("Predictive Modeling")
    meta = load_metadata()
    st.warning("PROVISIONAL model: this is not confirmed to be the model from your midterm. It estimates a derived target based on completed requests in the supplied dataset.")
    st.write("Target definition: **SLA breach = Processing_Days > SLA_Days** for completed requests with known processing time.")
    st.caption(f"Model version: {meta['model_version']} · Training rows: {meta['training_rows']} · Holdout rows: {meta['test_rows']}")
    with st.form("prediction_form"):
        left,right = st.columns(2)
        with left:
            service = st.selectbox("Service type", sorted(df["Service_Type"].dropna().astype(str).unique().tolist()) if "Service_Type" in df else ["Records", "Certification", "IT Support", "Scholarship Inquiry", "Enrollment"])
            requester = st.selectbox("Requester type", sorted(df["Requester_Type"].dropna().astype(str).unique().tolist()) if "Requester_Type" in df else ["Student", "Faculty", "Staff"])
            channel = st.selectbox("Requested channel", sorted(df["Requested_Channel"].dropna().astype(str).unique().tolist()) if "Requested_Channel" in df else ["Portal", "Email", "Walk-in"])
            priority = st.selectbox("Priority", ["Low", "Normal", "High"])
            office = st.selectbox("Assigned office", sorted(df["Assigned_Office"].dropna().astype(str).unique().tolist()) if "Assigned_Office" in df else ["Registrar", "IT Office", "Scholarship Office", "Admissions"])
        with right:
            queue = st.number_input("Current queue length", min_value=0, max_value=10000, value=10, step=1)
            documents = st.selectbox("Required documents complete?", ["Yes", "No"])
            sla = st.selectbox("SLA days", [3,5,7,10,14], index=2)
        submitted = st.form_submit_button("Estimate SLA-breach risk")
    if submitted:
        values = {
            "Service_Type": service, "Requester_Type": requester, "Requested_Channel": channel,
            "Priority": priority, "Assigned_Office": office, "Queue_Length": int(queue),
            "Required_Documents_Complete": documents, "SLA_Days": int(sla)
        }
        try:
            label, probability = predict_one(values)
            if label == 1:
                st.error("Model estimate: higher likelihood of an SLA breach.")
            else:
                st.success("Model estimate: predicted within SLA.")
            if probability is not None:
                st.metric("Estimated SLA-breach probability", f"{probability*100:.1f}%")
                st.progress(min(max(probability,0.0),1.0))
            st.json(values)
            st.info("Staff should verify current workload, documents, urgency, and office context before deciding what action to take.")
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")
    st.markdown("#### Holdout evaluation (provisional model)")
    metrics = meta["metrics"]
    a,b,c,d = st.columns(4)
    a.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    b.metric("Precision", f"{metrics['precision']:.3f}")
    c.metric("Recall", f"{metrics['recall']:.3f}")
    d.metric("F1 score", f"{metrics['f1']:.3f}")
    st.caption("These metrics are from a single stratified holdout split of a small synthetic-looking dataset; they are not evidence of real-world performance.")

    st.markdown("#### Five sample predictions (for demonstration)")
    st.write("Generate five predictions from example rows using only the model's allowed input features. These are demonstrations, not verified real-world outcomes.")
    if st.button("Generate 5 sample predictions"):
        sample_source = df.dropna(subset=meta["feature_columns"]).head(5).copy()
        if sample_source.empty:
            st.warning("The active dataset does not contain five usable example rows with all required features.")
        else:
            sample_model = __import__("modules.prediction", fromlist=["load_model"]).load_model()
            sample_X = sample_source[meta["feature_columns"]]
            sample_labels = sample_model.predict(sample_X)
            sample_probabilities = sample_model.predict_proba(sample_X)[:, list(sample_model.named_steps["classifier"].classes_).index(1)]
            sample_result = sample_source[[c for c in ["Request_ID", "Service_Type", "Priority", "Queue_Length", "SLA_Days"] if c in sample_source]].copy()
            sample_result["Predicted class"] = ["SLA breach risk" if int(v) == 1 else "Within SLA estimate" for v in sample_labels]
            sample_result["Estimated breach probability"] = [f"{float(v)*100:.1f}%" for v in sample_probabilities]
            st.dataframe(sample_result, use_container_width=True)
            st.caption("The predicted class is model output. Compare it with observed outcomes only where completed processing days are known.")

elif page == "Dashboard":
    render_dashboard(df)

elif page == "System Design":
    st.subheader("Systems Analysis and Design (SAD) Integration")
    st.markdown("#### Functional requirements")
    requirements = [
        "The system shall allow an authorized user to upload a CSV file and validate required columns.",
        "The system shall show a data preview, missing-value summary, and duplicate-request checks.",
        "The system shall display descriptive statistics and at least three interactive charts.",
        "The system shall accept request attributes and return a provisional SLA-breach class and probability when supported by the model.",
        "The system shall summarize request workload, identify pending requests for review, and provide human follow-up suggestions."
    ]
    for i, req in enumerate(requirements, 1):
        st.write(f"{i}. {req}")
    st.markdown("#### Non-functional requirements")
    st.write("- **Usability:** clear navigation, readable charts, and understandable messages.")
    st.write("- **Reliability:** validate uploaded schemas and handle invalid files without crashing the app.")
    st.write("- **Privacy and security:** use synthetic or authorized data, restrict access where required, and do not commit sensitive records.")
    st.markdown("#### Actors")
    st.write("- **Student Services Staff:** upload and inspect requests, review risk estimates, and make final decisions.")
    st.write("- **System Administrator / Analyst:** maintain the app, model artifact, schema, and deployment.")
    st.markdown("#### Use case diagram")
    image_path = ROOT / "assets" / "use_case_diagram.png"
    if image_path.exists():
        st.image(str(image_path), caption="Use case diagram — Student Service Request Monitoring")
    else:
        st.warning("Use case diagram image is missing from assets/.")
    st.markdown("#### Analytical process flow")
    st.code("CSV upload → schema validation → preview / null / duplicate checks → EDA and workload review → model risk estimate → staff checks context → authorized staff records final action", language="text")
    st.markdown("#### Model provenance and limitations")
    st.write("- Original notebook found: one code cell with synthetic-data generation and EDA.")
    st.write("- Notebook issue: variable `n` is referenced but not defined in the uploaded cell.")
    st.write("- No original target definition, training/evaluation code, or saved model was found in the provided notebook.")
    st.write("- Provisional target used here: `SLA_Breach = Processing_Days > SLA_Days`, for completed requests only.")
    st.write("- Predictors use fields available at request review: service, requester, channel, priority, assigned office, queue length, document completeness, and SLA days.")
    st.write("- `Processing_Days`, `Status`, `Satisfaction_Score`, and `Request_ID` are excluded from prediction features to reduce outcome leakage.")
    st.write("- The dataset appears synthetic and random; model scores may be weak and should not be presented as validated operational performance.")
