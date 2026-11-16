
import os

import cv2
import joblib
import numpy as np
import streamlit as st

from preprocessing import preprocess_array
from abnormality import analyze_image
from feature_extraction import extract_all_features


st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #172B3A;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.15rem !important;
        font-weight: 700 !important;
        color: #172B3A !important;
        letter-spacing: -0.01em;
        margin-bottom: 0.5rem;
    }

    h2 {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: #172B3A !important;
    }

    h3 {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #243B53 !important;
    }

    p, label, li {
        color: #243B53 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border: 1px solid #D9DEE2 !important;
        border-radius: 6px;
        box-shadow: none;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D9DEE2;
        border-radius: 6px;
        padding: 1rem 1.1rem;
    }

    div[data-testid="stMetricLabel"] p {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #334E68 !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: #172B3A !important;
    }

    div[data-testid="stImage"] img {
        border-radius: 4px;
    }

    div[data-testid="stImageCaption"] {
        color: #334E68 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }

    section[data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF;
        border: 1px dashed #AAB4BD;
        border-radius: 6px;
    }

    section[data-testid="stFileUploaderDropzone"] * {
        color: #243B53 !important;
        font-size: 1rem !important;
    }

    .result {
        border-radius: 6px;
        padding: 1.15rem 1.4rem;
        font-size: 1.5rem;
        font-weight: 700;
        border: 1px solid;
        border-left-width: 6px;
    }

    .result.normal {
        background: #EDF5F2;
        border-color: #2F6F5E;
        color: #1F4F43;
    }

    .result.pneumonia {
        background: #FAEEEE;
        border-color: #A33A3A;
        color: #7A2626;
    }

    .bar-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #243B53;
        margin-bottom: 0.5rem;
    }

    .bar-track {
        background: #E4E8EB;
        height: 13px;
        border-radius: 5px;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 5px;
    }

    .bar-fill.normal {
        background: #2F6F5E;
    }

    .bar-fill.pneumonia {
        background: #A33A3A;
    }

    [data-testid="stCaptionContainer"] p {
        font-size: 0.95rem !important;
        color: #52606D !important;
    }

    div[data-testid="stAlert"] p {
        font-size: 1rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #D9DEE2;
        border-radius: 6px;
    }

    div[data-testid="stExpander"] summary p {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #243B53 !important;
    }

    hr {
        margin: 1.5rem 0;
        border-color: #D9DEE2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Quantitative Image Analysis for Pneumonia Detection")

st.write(
    "Upload a chest X-ray to perform image preprocessing, "
    "lung segmentation, abnormality analysis and classification."
)

if not os.path.exists("BestModel.pkl"):
    st.error("BestModel.pkl not found. Run train_model.py first.")
    st.stop()

model_data = joblib.load("BestModel.pkl")
model = model_data["model"]
threshold = model_data["threshold"]
features = model_data["features"]
model_name = model_data["model_name"]

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    original_image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if original_image is None:
        st.error("Unable to read the uploaded image.")
        st.stop()

    processed = preprocess_array(original_image)

    analysis = analyze_image(processed)

    if analysis is None:
        st.error(
            "Could not find two clear lung fields in this image. "
            "Please upload a frontal chest X-ray."
        )
        st.stop()

    features_dict = extract_all_features(processed)

    if features_dict is None:
        st.error("Unable to extract features.")
        st.stop()

    x = np.array([[features_dict[f] for f in features]])
    p_pneumonia = float(model.predict_proba(x)[0, 1])
    is_pneumonia = p_pneumonia > threshold

    affected = analysis["affected_area_percentage"]

    MAX_RELIABLE = 60.0
    NORMAL_MAX = 25.0

    reliable = affected <= MAX_RELIABLE
    supports = (
        affected <= NORMAL_MAX
        if not is_pneumonia
        else affected > 5.0
    )

    overlay = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

    _layers = [(analysis["lung_outline"], (0, 255, 0))]

    if reliable:
        _layers.append((analysis["abnormal_mask"], (255, 0, 0)))

    for m, colour in _layers:
        cnts, _ = cv2.findContours(
            m,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, cnts, -1, colour, 1)

    st.divider()

    cls = "pneumonia" if is_pneumonia else "normal"
    label = "PNEUMONIA" if is_pneumonia else "NORMAL"

    left, right = st.columns([1, 2], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Original X-ray")

            st.image(
                cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

    with right:
        st.markdown(
            f'<div class="result {cls}">Prediction: {label}</div>',
            unsafe_allow_html=True,
        )

        st.caption(f"Classification model: {model_name}")

        st.subheader("Classification Probability")

        pct = min(max(p_pneumonia * 100, 0), 100)

        st.markdown(
            f'<div class="bar-label">Model Prediction Accuracy: '
            f'{p_pneumonia * 100:.2f}%</div>'
            f'<div class="bar-track"><div class="bar-fill {cls}" '
            f'style="width:{pct:.1f}%"></div></div>',
            unsafe_allow_html=True,
        )

        st.write("")

        if reliable and supports:
            m1, m2, m3 = st.columns(3)

            m1.metric("Lung Area", f"{analysis['lung_area']:,}")
            m2.metric("Abnormal Area", f"{analysis['abnormal_area']:,}")
            m3.metric("Affected Area", f"{affected:.2f}%")

        else:
            st.metric("Lung Area", f"{analysis['lung_area']:,}")

            st.info(
                "Affected-area estimate not shown: it could not be measured "
                "reliably for this image. The prediction above is based on the "
                "full set of image features."
            )

    st.subheader("Analysis")

    c1, c2, c3, c4 = st.columns(4, gap="small")

    with c1:
        with st.container(border=True):
            st.image(
                processed,
                caption="Preprocessed X-ray",
                use_container_width=True
            )

    with c2:
        with st.container(border=True):
            st.image(
                analysis["lung_mask"],
                caption="Analysed lung field",
                use_container_width=True
            )

    with c3:
        with st.container(border=True):
            if reliable:
                st.image(
                    analysis["abnormal_mask"],
                    caption="Abnormal region",
                    use_container_width=True
                )
            else:
                st.write(
                    "Abnormal region not shown "
                    "(measurement unreliable for this image)"
                )

    with c4:
        with st.container(border=True):
            st.image(
                overlay,
                caption="Overlay",
                use_container_width=True
            )

    st.write("")

    with st.expander("Quantitative Features"):
        st.dataframe(
            features_dict,
            use_container_width=True
        )

    st.caption(
        "This system is an academic image-analysis project and is not a "
        "medical diagnostic tool."
    )
