
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="OptiMed-Fusion | AI Research Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0B1120;
    color: #E5E7EB;
}

[data-testid="stSidebar"] {
    background: #101A2D;
    border-right: 1px solid #24334B;
}

[data-testid="stSidebar"] * {
    color: #DCE6F5;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.hero {
    background: linear-gradient(120deg, #14243D, #102F3B);
    border: 1px solid #28536A;
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0;
}

.hero p {
    color: #A9C2D6;
    margin-top: 10px;
    font-size: 1rem;
}

.eyebrow {
    color: #40D6C2;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.metric-card {
    background: #131F33;
    border: 1px solid #273B55;
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #91A5BE;
    font-size: 0.85rem;
    font-weight: 500;
}

.metric-value {
    color: #F8FAFC;
    font-size: 1.9rem;
    font-weight: 800;
    margin-top: 8px;
}

.metric-note {
    color: #40D6C2;
    font-size: 0.75rem;
    margin-top: 4px;
}

.section-heading {
    font-size: 1.35rem;
    font-weight: 700;
    color: #F1F5F9;
    margin: 22px 0 12px 0;
}

.info-panel {
    background: #131F33;
    border: 1px solid #273B55;
    padding: 19px;
    border-radius: 14px;
    margin-bottom: 12px;
}

.info-panel h4 {
    color: #40D6C2;
    margin-bottom: 8px;
}

.info-panel p {
    color: #A9B9CE;
    margin-bottom: 0;
}

.small-muted {
    color: #8FA3BC;
    font-size: 0.82rem;
}

div.stButton > button[kind="primary"],
div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #0D9488, #0F766E);
    border: none;
    color: white;
    border-radius: 9px;
    font-weight: 700;
    min-height: 45px;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    border: 1px solid #40D6C2;
    color: white;
}

hr {
    border-color: #26364D;
}

[data-testid="stMetric"] {
    background: #131F33;
    border: 1px solid #273B55;
    padding: 15px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #A9B9CE;
}

[data-testid="stMetricValue"] {
    color: #F8FAFC;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# REUSABLE UI FUNCTIONS
# =========================================================

def hero(title, subtitle, eyebrow="OPTIMED-FUSION RESEARCH PLATFORM"):
    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, note=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def section(title):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )


def styled_chart(fig):
    fig.update_layout(
        paper_bgcolor="#131F33",
        plot_bgcolor="#131F33",
        font=dict(color="#DCE6F5"),
        margin=dict(l=25, r=25, t=45, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#DCE6F5")
        )
    )
    return fig


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🧬 OptiMed")
    st.markdown("### FUSION PLATFORM")
    st.caption("Multimodal AI Research System")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Patient Assessment",
            "Fusion Analysis",
            "Model Insights",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("**SYSTEM STATUS**")
    st.markdown("🟢 Local inference pipeline verified")
    st.markdown("🟢 XGBoost JSON model")
    st.markdown("🟢 Random Forest model")
    st.markdown("🟢 CNN model")

    st.divider()

    st.caption("Version 1.0.0")
    st.caption("Synthetic Research Prototype")


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    hero(
        "OptiMed-Fusion",
        "A multimodal machine learning framework combining "
        "clinical tabular features and grayscale image analysis."
    )

    st.warning(
        "Research prototype only. All project data are synthetic. "
        "This system is not clinically validated and must not be "
        "used for diagnosis or medical decisions."
    )

    section("Platform Overview")

    cols = st.columns(4)

    with cols[0]:
        metric_card("Synthetic Patients", "1,500", "Research dataset")

    with cols[1]:
        metric_card("Clinical Features", "06", "Tabular inputs")

    with cols[2]:
        metric_card("Model Branches", "02", "Tabular + Image")

    with cols[3]:
        metric_card("Fusion Strategy", "65 : 35", "ML + CNN weights")

    section("Multimodal Architecture")

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="info-panel">
            <h4>01 · Clinical Intelligence</h4>
            <p>Six clinical features → StandardScaler → XGBoost
            + Random Forest → Soft Voting Ensemble</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="info-panel">
            <h4>02 · Image Intelligence</h4>
            <p>32 × 32 grayscale scan → Convolutional Neural
            Network → Image probability</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-panel">
        <h4>03 · Weighted Late Fusion</h4>
        <p>Tabular ML probability (65%) + CNN probability (35%)
        → Final model output → Research risk category</p>
    </div>
    """, unsafe_allow_html=True)

    section("Research Workflow")

    workflow = [
        ("01", "Data Generation", "Synthetic paired patient dataset"),
        ("02", "Preprocessing", "Scaling and training-only SMOTE"),
        ("03", "Model Training", "Optuna XGBoost, Random Forest and CNN"),
        ("04", "Ensemble Learning", "Soft voting and late fusion"),
        ("05", "Evaluation", "Classification metrics and ROC analysis"),
        ("06", "Inference", "Local multimodal prediction pipeline")
    ]

    for number, title, description in workflow:
        st.markdown(f"""
        <div class="info-panel">
            <h4>{number} · {title}</h4>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PATIENT ASSESSMENT
# =========================================================

elif page == "Patient Assessment":

    hero(
        "Patient Assessment",
        "Interactive inference interface for the trained multimodal models.",
        "LIVE MODEL INFERENCE"
    )

    st.warning(
        "Use only synthetic demonstration inputs and synthetic images. "
        "Uploaded images are not clinically interpreted."
    )

    section("Clinical Feature Configuration")

    with st.form("assessment_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=1.0,
                max_value=120.0,
                value=45.0
            )

            systolic_bp = st.number_input(
                "Systolic Blood Pressure",
                min_value=1.0,
                value=120.0
            )

            diastolic_bp = st.number_input(
                "Diastolic Blood Pressure",
                min_value=1.0,
                value=80.0
            )

        with col2:
            glucose = st.number_input(
                "Glucose",
                min_value=0.0,
                value=100.0
            )

            cholesterol = st.number_input(
                "Cholesterol",
                min_value=0.0,
                value=180.0
            )

            biomarker_score = st.number_input(
                "Biomarker Score",
                value=0.5,
                format="%.5f"
            )

        st.divider()

        section("Grayscale Scan Input")

        uploaded_scan = st.file_uploader(
            "Upload synthetic demonstration image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_scan:
            st.image(
                uploaded_scan,
                caption="Uploaded image preview",
                width=220
            )

        submitted = st.form_submit_button(
            "Run Multimodal Inference",
            use_container_width=True
        )

    if submitted:

        if uploaded_scan is None:
            st.error("Please upload an image.")

        else:
            try:
                from src.predict import predict_patient
                from src.image_preprocessing import preprocess_scan

                clinical_features = {
                    "age": age,
                    "systolic_bp": systolic_bp,
                    "diastolic_bp": diastolic_bp,
                    "glucose": glucose,
                    "cholesterol": cholesterol,
                    "biomarker_score": biomarker_score
                }

                scan_array = preprocess_scan(uploaded_scan)

                with st.spinner("Executing XGBoost, RF and CNN inference..."):

                    result = predict_patient(
                        clinical_features,
                        scan_array
                    )

                st.session_state["assessment_result"] = result

                st.success("Model inference completed.")

            except Exception as error:
                st.error(f"Inference error: {error}")

    if "assessment_result" in st.session_state:

        result = st.session_state["assessment_result"]

        st.divider()
        section("Individual Model Outputs")

        cols = st.columns(3)

        model_results = [
            ("XGBoost", result["xgboost_probability"]),
            ("Random Forest", result["random_forest_probability"]),
            ("CNN", result["cnn_probability"])
        ]

        for col, (name, probability) in zip(cols, model_results):
            with col:
                st.metric(name, f"{probability * 100:.2f}%")

        section("Weighted Fusion Output")

        cols = st.columns(2)

        cols[0].metric(
            "Tabular Ensemble",
            f"{result['ml_probability'] * 100:.2f}%"
        )

        cols[1].metric(
            "Final Fusion Output",
            f"{result['final_probability'] * 100:.2f}%"
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["final_probability"] * 100,
            number={"suffix": "%"},
            title={"text": "Final Model Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#40D6C2"},
                "bgcolor": "#131F33",
                "borderwidth": 1
            }
        ))

        fig = styled_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"Model output category: {result['risk_zone']}"
        )

        st.caption(
            "This is a synthetic-model output, not a clinical risk estimate."
        )


# =========================================================
# FUSION ANALYSIS
# =========================================================

elif page == "Fusion Analysis":

    hero(
        "Fusion Analysis",
        "Understand how independent model branches contribute "
        "to the final weighted prediction.",
        "MODEL INTERPRETABILITY"
    )

    section("Late Fusion Formula")

    st.latex(
        r"P_{final}=0.65P_{ML}+0.35P_{CNN}"
    )

    st.markdown("""
    <div class="info-panel">
        <h4>Tabular Branch · 65%</h4>
        <p>XGBoost and Random Forest probabilities are combined
        using soft voting.</p>
    </div>
    <div class="info-panel">
        <h4>Image Branch · 35%</h4>
        <p>The CNN independently extracts features from the
        grayscale scan and produces an image probability.</p>
    </div>
    """, unsafe_allow_html=True)

    if "assessment_result" in st.session_state:

        result = st.session_state["assessment_result"]

        section("Latest Assessment Breakdown")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=["Tabular ML", "CNN", "Final Fusion"],
            y=[
                result["ml_probability"] * 100,
                result["cnn_probability"] * 100,
                result["final_probability"] * 100
            ],
            text=[
                f"{result['ml_probability']*100:.2f}%",
                f"{result['cnn_probability']*100:.2f}%",
                f"{result['final_probability']*100:.2f}%"
            ],
            textposition="outside"
        ))

        fig.update_layout(
            title="Probability Comparison",
            yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 110])
        )

        st.plotly_chart(
            styled_chart(fig),
            use_container_width=True
        )

    else:
        st.info(
            "Run an assessment first to view actual fusion outputs here."
        )

    section("Fusion Design")

    st.write(
        "The current implementation uses fixed weights of 0.65 and 0.35. "
        "These are configured design weights, not learned or calibrated "
        "probability weights."
    )


# =========================================================
# MODEL INSIGHTS
# =========================================================

elif page == "Model Insights":

    hero(
        "Model Insights",
        "Evaluation artifacts, model comparison and research performance.",
        "EXPERIMENTAL ANALYTICS"
    )

    section("Model Evaluation")

    metrics_file = REPORTS_DIR / "model_performance_comparison.csv"

    if metrics_file.exists():

        metrics_df = pd.read_csv(metrics_file)

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "Model comparison CSV was not found in reports/."
        )

    section("Experiment Visualizations")

    figure_files = [
        ("Model Performance Comparison", "model_performance_comparison.png"),
        ("ROC Curve Comparison", "roc_curve_comparison.png"),
        ("Final Confusion Matrix", "final_confusion_matrix.png"),
        ("CNN Training History", "cnn_training_history.png")
    ]

    for title, filename in figure_files:

        image_path = FIGURES_DIR / filename

        if image_path.exists():
            with st.expander(title, expanded=False):
                st.image(
                    str(image_path),
                    use_container_width=True
                )

    section("Evaluation Methodology")

    st.markdown("""
    - Accuracy
    - Precision
    - Recall
    - Macro F1-score
    - ROC-AUC
    - Confusion Matrix
    - Classification Report
    """)

    st.caption(
        "All reported results belong to experiments on synthetic data "
        "and should not be interpreted as clinical performance."
    )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "About Project":

    hero(
        "About OptiMed-Fusion",
        "A research-oriented multimodal machine learning project.",
        "PROJECT DOCUMENTATION"
    )

    section("Technical Stack")

    stack = {
        "Language": "Python",
        "ML Framework": "Scikit-learn, XGBoost",
        "Deep Learning": "TensorFlow / Keras",
        "Optimization": "Optuna TPE",
        "Imbalance Handling": "SMOTE",
        "Visualization": "Plotly, Matplotlib",
        "Application": "Streamlit",
        "Serialization": "Joblib, XGBoost JSON"
    }

    st.table(
        pd.DataFrame(
            list(stack.items()),
            columns=["Component", "Technology"]
        )
    )

    section("Dataset Description")

    st.markdown("""
    - Total synthetic paired records: 1,500
    - Healthy class: 1,200
    - High-risk class: 300
    - Clinical features: 6 continuous variables
    - Image dimensions: 32 × 32 × 1
    - Stratified train-validation split: 80:20
    """)

    section("Research Limitations")

    st.warning("""
    This project uses entirely synthetic data. Its outputs have not
    been validated on real patients, medical imaging datasets or
    clinical workflows. The displayed categories are experimental
    model outputs and must not guide diagnosis or treatment.
    """)

    section("Project Objective")

    st.write(
        "The objective is to demonstrate multimodal machine learning, "
        "imbalanced classification, hyperparameter optimization, "
        "CNN-based image processing, ensemble learning, late fusion "
        "and deployment-oriented engineering."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px;">
    <p style="color:#40D6C2; font-weight:700;">
        OPTIMED-FUSION · MULTIMODAL AI RESEARCH
    </p>
    <p style="color:#8296AF; font-size:0.8rem;">
        Synthetic Research Prototype · Not for Clinical Use
    </p>
</div>
""", unsafe_allow_html=True)
