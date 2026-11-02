"""Interactive Streamlit dashboard for customer churn analysis and prediction."""

import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from data_preprocessing import clean_dataset
from evaluate import main as evaluate_models
from train import main as train_models


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_customer_churn_clean.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"

st.set_page_config(
    page_title="Customer Churn | Insight Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html(
    """
    <script>
    (() => {
        const app = document.querySelector('[data-testid="stApp"]');
        if (!app || app.dataset.themeObserverAttached) return;

        const updateTheme = () => {
            const channels = getComputedStyle(document.body).backgroundColor
                .match(/\\d+/g)?.slice(0, 3).map(Number);
            const isDark = channels && channels.reduce((sum, value) => sum + value, 0) < 384;
            app.dataset.streamlitTheme = isDark ? "dark" : "light";
        };

        updateTheme();
        new MutationObserver(updateTheme).observe(document.head, {
            childList: true,
            characterData: true,
            subtree: true,
        });
        new MutationObserver(updateTheme).observe(document.body, {
            attributes: true,
            attributeFilter: ["class", "style"],
        });
        app.dataset.themeObserverAttached = "true";
    })();
    </script>
    """,
    unsafe_allow_javascript=True,
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f5f8fc 0%, #fbfcfe 38%, #ffffff 100%);
    }
    [data-testid="stAppViewContainer"] {
        color: #172941 !important;
        color-scheme: light;
    }
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h5,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h6,
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] label p,
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stTab"],
    [data-testid="stAppViewContainer"] [data-testid="stTab"] p {
        color: #172941 !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p,
    [data-testid="stAppViewContainer"] .section-note {
        color: #66758a !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stTab"][aria-selected="false"],
    [data-testid="stAppViewContainer"] [data-testid="stTab"][aria-selected="false"] p {
        color: #52647a !important;
    }
    #stMain .hero h1 { color: #ffffff !important; }
    #stMain .hero p { color: #e0edf7 !important; }
    #stMain .hero .hero-eyebrow { color: #b7d9ee !important; }
    #stMain .section-note { color: #66758a !important; }
    [data-testid="stHeader"] { background: rgba(245, 248, 252, 0.88); }
    [data-testid="stSidebar"] {
        background: #101b31;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] * { color: #e8eef8; }
    [data-testid="stSidebar"] code {
        color: #f3f7fc;
        background: rgba(255, 255, 255, 0.12);
    }
    .hero {
        padding: 1.65rem 1.8rem;
        border-radius: 20px;
        color: #ffffff;
        background: linear-gradient(115deg, #142747 0%, #1d4778 58%, #287d91 100%);
        margin: 0.25rem 0 1.35rem 0;
        box-shadow: 0 14px 34px rgba(27, 62, 103, 0.14);
    }
    .hero-eyebrow {
        margin: 0 0 0.35rem 0;
        color: #b7d9ee;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .hero h1 { margin: 0; color: #ffffff !important; font-size: 2rem; }
    .hero p { margin: 0.45rem 0 0 0; color: #e0edf7 !important; }
    .section-heading {
        color: #172941;
        font-size: 1.22rem;
        font-weight: 700;
        margin: 0.5rem 0 0.15rem 0;
    }
    .section-note { color: #66758a; font-size: 0.9rem; margin-bottom: 0.7rem; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4eaf2;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        box-shadow: 0 5px 16px rgba(29, 55, 88, 0.045);
    }
    label[data-testid="stMetricLabel"],
    label[data-testid="stMetricLabel"] p { color: #65748a !important; }
    div[data-testid="stMetricValue"] { color: #172941; }
    .model-name-card {
        background: #ffffff;
        border: 1px solid #e4eaf2;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        box-shadow: 0 5px 16px rgba(29, 55, 88, 0.045);
        min-height: 7.8rem;
        box-sizing: border-box;
    }
    .model-name-label {
        color: #65748a;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }
    .model-name-value {
        color: #172941;
        font-size: clamp(1.2rem, 2vw, 2rem);
        line-height: 1.2;
        overflow-wrap: anywhere;
    }
    .stApp[data-streamlit-theme="dark"] {
        background: linear-gradient(180deg, #0e1117 0%, #111827 38%, #151b26 100%);
    }
    .stApp[data-streamlit-theme="dark"] [data-testid="stAppViewContainer"],
    .stApp[data-streamlit-theme="dark"] [data-testid="stMarkdownContainer"],
    .stApp[data-streamlit-theme="dark"] [data-testid="stMarkdownContainer"] p,
    .stApp[data-streamlit-theme="dark"] [data-testid="stMarkdownContainer"] h1,
    .stApp[data-streamlit-theme="dark"] [data-testid="stMarkdownContainer"] h2,
    .stApp[data-streamlit-theme="dark"] [data-testid="stMarkdownContainer"] h3,
    .stApp[data-streamlit-theme="dark"] [data-testid="stMarkdownContainer"] h4,
    .stApp[data-streamlit-theme="dark"] [data-testid="stMarkdownContainer"] h5,
    .stApp[data-streamlit-theme="dark"] [data-testid="stMarkdownContainer"] h6,
    .stApp[data-streamlit-theme="dark"] [data-testid="stWidgetLabel"],
    .stApp[data-streamlit-theme="dark"] [data-testid="stWidgetLabel"] p,
    .stApp[data-streamlit-theme="dark"] label,
    .stApp[data-streamlit-theme="dark"] label p,
    .stApp[data-streamlit-theme="dark"] [data-testid="stTab"] {
        color: #f0f2f6 !important;
    }
    .stApp[data-streamlit-theme="dark"] [data-testid="stCaptionContainer"],
    .stApp[data-streamlit-theme="dark"] [data-testid="stCaptionContainer"] p,
    .stApp[data-streamlit-theme="dark"] .section-note {
        color: #aab6c5 !important;
    }
    .stApp[data-streamlit-theme="dark"] [data-testid="stTab"][aria-selected="false"],
    .stApp[data-streamlit-theme="dark"] [data-testid="stTab"][aria-selected="false"] p {
        color: #b6c2d1 !important;
    }
    .stApp[data-streamlit-theme="dark"] [data-testid="stHeader"] {
        background: rgba(14, 17, 23, 0.88);
    }
    .stApp[data-streamlit-theme="dark"] div[data-testid="stMetric"],
    .stApp[data-streamlit-theme="dark"] .model-name-card {
        background: #1b2533;
        border-color: #344154;
    }
    .stApp[data-streamlit-theme="dark"] label[data-testid="stMetricLabel"],
    .stApp[data-streamlit-theme="dark"] label[data-testid="stMetricLabel"] p,
    .stApp[data-streamlit-theme="dark"] .model-name-label { color: #aab6c5 !important; }
    .stApp[data-streamlit-theme="dark"] div[data-testid="stMetricValue"],
    .stApp[data-streamlit-theme="dark"] .model-name-value { color: #f0f2f6 !important; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        color: white;
        background: #1d6383;
        border: 0;
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 700;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        color: white;
        background: #174e6a;
        border-color: #174e6a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_artifact(model_path: str) -> dict:
    """Train the model on first deployment, then load and cache the artifact."""
    model_file = Path(model_path)
    if not model_file.exists():
        train_models()
        evaluate_models()
    return joblib.load(model_path)


@st.cache_data
def load_customer_data(data_path: str, modified_time: float) -> pd.DataFrame:
    """Load and validate the available local dataset for dashboard statistics."""
    del modified_time  # Include the file timestamp in the cache key for refreshes.
    return clean_dataset(pd.read_csv(data_path))


def build_customer_record(values: dict[str, object]) -> pd.DataFrame:
    """Create the one-row input frame with the model's training feature names."""
    return pd.DataFrame([values])


with st.spinner("Preparing the churn model on first startup..." if not MODEL_PATH.exists() else "Loading churn model..."):
    artifact = load_model_artifact(str(MODEL_PATH))
model = artifact["model"]

st.markdown(
    """
    <div class="hero">
        <p class="hero-eyebrow">Customer intelligence · Telco</p>
        <h1>Churn insights, made actionable</h1>
        <p>Explore customer retention patterns and estimate churn risk for an individual account.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 📊 Retention studio")
    st.caption("CUSTOMER ANALYTICS")
    st.markdown("---")
    st.markdown("**Active model**")
    st.markdown(f"`{artifact['model_name']}`")
    if "metrics" in artifact:
        st.caption(f"Holdout churn F1-score: {artifact['metrics']['f1_score']:.1%}")
    st.markdown("---")
    st.markdown("**About this dashboard**")
    st.caption(
        "Built with the public IBM Telco Customer Churn sample. "
        "Model predictions are estimates, not guarantees."
    )

if PROCESSED_DATA_PATH.exists():
    DATA_PATH = PROCESSED_DATA_PATH
elif RAW_DATA_PATH.exists():
    DATA_PATH = RAW_DATA_PATH
else:
    st.error(
        "The customer dataset is missing. Run `python data_preprocessing.py` "
        "or `python train.py` to prepare it."
    )
    st.stop()

customer_data = load_customer_data(
    str(DATA_PATH),
    DATA_PATH.stat().st_mtime,
)
if customer_data.empty:
    st.error("The dataset has no valid customer records to display.")
    st.stop()

overview_tab, prediction_tab = st.tabs(["📈  Dashboard overview", "🔎  Predict a customer"])

with overview_tab:
    total_customers = len(customer_data)
    churned_customers = int((customer_data["Churn"] == "Yes").sum())
    churn_rate = churned_customers / total_customers
    stayed_customers = total_customers - churned_customers
    churned_monthly_charge = customer_data.loc[
        customer_data["Churn"] == "Yes", "MonthlyCharges"
    ].mean()

    st.markdown('<p class="section-heading">Portfolio snapshot</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-note">A high-level view of the customer base in the training dataset.</p>',
        unsafe_allow_html=True,
    )
    total_col, churned_col, rate_col, charge_col = st.columns(4)
    total_col.metric("Customers analyzed", f"{total_customers:,}")
    churned_col.metric("Customers churned", f"{churned_customers:,}")
    rate_col.metric("Overall churn rate", f"{churn_rate:.1%}")
    charge_col.metric(
        "Avg. monthly charge · churned",
        f"${churned_monthly_charge:,.2f}" if pd.notna(churned_monthly_charge) else "N/A",
    )

    chart_col, contract_col = st.columns([0.85, 1.4], gap="large")
    with chart_col:
        st.markdown('<p class="section-heading">Customer outcomes</p>', unsafe_allow_html=True)
        outcome_counts = pd.DataFrame(
            {"Customers": [stayed_customers, churned_customers]},
            index=["Stayed", "Churned"],
        )
        st.bar_chart(outcome_counts, color=["#2a9d8f"], height=290)
        st.caption("Count of customers by observed churn outcome.")

    with contract_col:
        st.markdown('<p class="section-heading">Churn by contract</p>', unsafe_allow_html=True)
        contract_rates = (
            pd.crosstab(customer_data["Contract"], customer_data["Churn"], normalize="index")
            .mul(100)
            .reindex(columns=["No", "Yes"], fill_value=0)
            .rename(columns={"No": "Stayed", "Yes": "Churned"})
        )
        st.bar_chart(
            contract_rates,
            stack=True,
            color=["#87b8c5", "#e47765"],
            height=290,
        )
        st.caption("Share of each contract group that stayed or churned.")

    st.markdown('<p class="section-heading">Tenure and retention</p>', unsafe_allow_html=True)
    tenure_bins = [0, 12, 24, 48, 72, float("inf")]
    tenure_labels = ["0–12 months", "13–24 months", "25–48 months", "49–72 months", "73+ months"]
    tenure_data = customer_data.assign(
        TenureGroup=pd.cut(
            customer_data["tenure"],
            bins=tenure_bins,
            labels=tenure_labels,
            include_lowest=True,
        )
    )
    tenure_churn = (
        tenure_data.groupby("TenureGroup", observed=False)["Churn"]
        .apply(lambda churn: (churn == "Yes").mean() * 100)
        .reindex(tenure_labels)
        .rename("Churn rate (%)")
    )
    st.bar_chart(tenure_churn, color="#e47765", height=270)
    st.caption("Observed churn rate by customer tenure. Smaller tenure cohorts can vary.")

    st.info(
        "These charts summarize the public sample dataset, not live business data. "
        "Use the prediction tab to estimate risk for one customer."
    )

with prediction_tab:
    intro_col, model_col = st.columns([3, 1])
    with intro_col:
        st.markdown('<p class="section-heading">Customer information</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-note">Enter account details to estimate the chance of churn.</p>',
            unsafe_allow_html=True,
        )
    with model_col:
        st.markdown(
            '<div class="model-name-card">'
            '<div class="model-name-label">Model in use</div>'
            f'<div class="model-name-value">{html.escape(artifact["model_name"])}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with st.form("customer_details"):
        st.markdown("##### Profile and account")
        profile_left, profile_right = st.columns(2)
        with profile_left:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda value: "Yes" if value else "No",
            )
            partner = st.selectbox("Has a partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            tenure = st.number_input(
                "Months with the company", min_value=0, max_value=100, value=12
            )
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online backup", ["Yes", "No", "No internet service"]
            )
        with profile_right:
            device_protection = st.selectbox(
                "Device protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming movies", ["Yes", "No", "No internet service"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input(
                "Monthly charges ($)", min_value=0.0, value=70.0, step=1.0
            )
            total_charges = st.number_input(
                "Total charges ($)", min_value=0.0, value=840.0, step=10.0
            )

        submitted = st.form_submit_button(
            "Predict churn risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        customer = build_customer_record(
            {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        )
        prediction = model.predict(customer)[0]
        probabilities = model.predict_proba(customer)[0]
        churn_probability = float(probabilities[list(model.classes_).index("Yes")])

        st.markdown("---")
        result_col, probability_col = st.columns([1.2, 1], gap="large")
        with result_col:
            st.markdown('<p class="section-heading">Prediction result</p>', unsafe_allow_html=True)
            if prediction == "Yes":
                st.warning("Higher churn risk — this customer is predicted to churn.")
            else:
                st.success("Lower churn risk — this customer is predicted to stay.")
            st.caption(
                f"Prediction from the {artifact['model_name']} model. "
                "This estimate is not a guarantee."
            )
        with probability_col:
            st.metric("Estimated churn probability", f"{churn_probability:.1%}")
            st.progress(churn_probability, text="Churn probability")
