# Section B — Student Service Request Monitoring (Streamlit)

A Streamlit decision-support app for the Section B laboratory exercise. It includes Home, Dataset, Data Analysis, Prediction, Dashboard, and System Design pages.

## Important: model provenance

The supplied notebook `SectionB_DataMining_Exam (2).ipynb` contains one code cell for synthetic data generation and exploratory analysis. It refers to `n` without defining it and contains no fitted model, target definition, model evaluation, or saved model. Therefore, the app includes a **PROVISIONAL** model so that the Prediction module is functional. It must not be described as the original midterm model.

The provisional target is derived from the provided CSV: `SLA_Breach = 1` when `Processing_Days > SLA_Days`, otherwise `0`, and only completed requests with known processing time are used to train/evaluate it. Predictors are `Service_Type`, `Requester_Type`, `Requested_Channel`, `Priority`, `Assigned_Office`, `Queue_Length`, `Required_Documents_Complete`, and `SLA_Days`. Future/outcome fields such as `Processing_Days`, `Status`, and `Satisfaction_Score` are excluded from prediction features.

The CSV has 400 rows and 14 columns; 100 `Processing_Days` and 100 `Satisfaction_Score` values are missing because non-completed requests have no observed result. The data appears synthetic/random. The model's metrics are a simple holdout estimate on a small dataset, not proof of real-world predictive performance.

## Modules

1. **Dataset** — upload CSV, required-column validation, data preview, null and duplicate summary, download active data.
2. **Data Analysis** — descriptive statistics and interactive charts for request categories, queue/SLA relationship, processing days, and observed SLA outcomes.
3. **Prediction** — form for a single hypothetical request, predicted class and probability (when supported), model metrics and limitations.
4. **Dashboard** — workload KPIs, observed SLA breach rate, pending-request risk queue, human follow-up suggestions.
5. **System Design** — five functional requirements, three non-functional requirements, actors, use case diagram, analytical workflow, provenance notes.

## Run locally

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The bundled model is in `models/trained_model.joblib`; its provenance and holdout metrics are in `models/model_metadata.json`. The sample dataset is in `data/student_service_requests.csv`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload all files/folders in this project.
2. Open https://share.streamlit.io and sign in.
3. Create an app from your repository, choose the branch, and set `app.py` as the main file.
4. Deploy, then test each page and upload a valid and invalid CSV.
5. Save the repository URL and deployed app URL for your submission.

## Testing checklist

- [ ] Home page opens and shows provisional-model notice.
- [ ] Valid sample CSV passes validation.
- [ ] CSV missing required columns shows an error and does not replace active data.
- [ ] Missing values and duplicate IDs are summarized.
- [ ] Data Analysis displays descriptive statistics and interactive charts.
- [ ] Prediction returns a class and probability for a valid form.
- [ ] Dashboard shows KPIs and a review queue when pending requests exist.
- [ ] System Design shows requirements, actors, use case diagram, workflow, and limitations.
- [ ] No real student-sensitive data is deployed publicly.
- [ ] Replace the provisional model with the original midterm pipeline if it is later located; then verify predictions match the midterm results.

## Submission report starter

**Reused code:** the uploaded notebook's dataset fields and EDA context were reviewed; the notebook's synthetic data-generation/EDA code was not directly reusable as a working model pipeline because `n` is undefined and no model was saved.

**Changes:** built a six-page Streamlit interface; added schema validation, null/duplicate reporting, interactive EDA, a provisional SLA-breach model, dashboard risk review, SAD requirements and diagram.

**Testing:** run through the checklist above and capture screenshots of Dataset, Data Analysis, Prediction, Dashboard, System Design, and five sample predictions.

**Limitations:** the model is provisional, the dataset appears synthetic/random, the target is derived rather than recovered from midterm instructions, and model metrics are not validated for operational decisions. Human review is mandatory.
