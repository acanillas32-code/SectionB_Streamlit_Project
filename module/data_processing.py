import pandas as pd

REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Required_Documents_Complete", "Processing_Days", "SLA_Days",
    "Status", "Reopened", "Satisfaction_Score"
]

FEATURE_COLUMNS = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Queue_Length", "Required_Documents_Complete", "SLA_Days"
]

def validate_dataset(df):
    errors = []
    if df is None or df.empty:
        return ["The CSV has no rows."]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
    if df.columns.duplicated().any():
        errors.append("Duplicate column names were found.")
    if "Request_ID" in df.columns and df["Request_ID"].duplicated().any():
        errors.append(f"Duplicate Request_ID values: {int(df['Request_ID'].duplicated().sum())}.")
    for col in ["Queue_Length", "SLA_Days", "Processing_Days", "Satisfaction_Score"]:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            invalid = df[col].notna() & converted.isna()
            if invalid.any():
                errors.append(f"{col} contains {int(invalid.sum())} non-numeric value(s).")
    if "SLA_Days" in df.columns and (pd.to_numeric(df["SLA_Days"], errors="coerce").dropna() <= 0).any():
        errors.append("SLA_Days must be greater than zero.")
    return errors

def clean_for_display(df):
    result = df.copy()
    if "Submission_Date" in result.columns:
        result["Submission_Date"] = pd.to_datetime(result["Submission_Date"], errors="coerce")
    for col in ["Queue_Length", "SLA_Days", "Processing_Days", "Satisfaction_Score"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result

def add_observed_sla_breach(df):
    """Only completed requests with known processing days have an observed outcome."""
    result = df.copy()
    result["SLA_Breach"] = pd.NA
    if {"Status", "Processing_Days", "SLA_Days"}.issubset(result.columns):
        done = result["Status"].eq("Completed") & result["Processing_Days"].notna() & result["SLA_Days"].notna()
        result.loc[done, "SLA_Breach"] = (
            pd.to_numeric(result.loc[done, "Processing_Days"], errors="coerce")
            > pd.to_numeric(result.loc[done, "SLA_Days"], errors="coerce")
        ).astype(int)
    return result
