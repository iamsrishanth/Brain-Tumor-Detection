"""
Brain Tumor MRI Classifier — Streamlit Web App
===============================================
Upload an MRI brain scan and get instant classification:
  - Glioma Tumor
  - Meningioma Tumor
  - No Tumor
  - Pituitary Tumor

Usage:
    streamlit run app.py
"""

import os
import subprocess
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# Pull latest changes before running
_repo_dir = os.path.dirname(os.path.abspath(__file__))
print("Pulling latest changes from git...")
try:
    result = subprocess.run(
        ["git", "pull"], cwd=_repo_dir, capture_output=True, text=True, timeout=30
    )
    print(result.stdout.strip() if result.stdout.strip() else "Already up to date.")
    if result.stderr and "error" in result.stderr.lower():
        print(f"Git warning: {result.stderr.strip()}")
except Exception as e:
    print(f"Git pull skipped: {e}")
print()

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_SIZE = 300
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Human-readable labels
CLASS_LABELS = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

# Color coding for results
CLASS_COLORS = {
    "glioma": "#e74c3c",
    "meningioma": "#e67e22",
    "notumor": "#27ae60",
    "pituitary": "#8e44ad",
}

# Tumor descriptions
TUMOR_INFO = {
    "glioma": {
        "description": "Gliomas are tumors that arise from glial cells (supportive tissue) of the brain. They are the most common type of primary brain tumor and can be low-grade (slow-growing) or high-grade (aggressive).",
        "severity": "High",
        "action": "Immediate consultation with a neuro-oncologist is recommended.",
    },
    "meningioma": {
        "description": "Meningiomas develop from the meninges — the membranes surrounding the brain and spinal cord. Most are benign (non-cancerous) and slow-growing.",
        "severity": "Low to Moderate",
        "action": "Regular monitoring and consultation with a neurosurgeon advised.",
    },
    "notumor": {
        "description": "No tumor detected in this MRI scan. The brain appears normal with no abnormal growths identified.",
        "severity": "None",
        "action": "No immediate action required. Continue regular health check-ups.",
    },
    "pituitary": {
        "description": "Pituitary tumors form in the pituitary gland at the base of the brain. Most are benign and can affect hormone production.",
        "severity": "Low to Moderate",
        "action": "Endocrinologist consultation recommended for hormone level assessment.",
    },
}


@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)."""
    model_path = os.path.join(MODEL_DIR, "best_model.keras")
    if not os.path.exists(model_path):
        st.error(
            f"Model not found at `{model_path}`. "
            "Please run `python train.py` first to train the model."
        )
        st.stop()

    model = tf.keras.models.load_model(model_path)

    # Load class labels if available
    labels_path = os.path.join(MODEL_DIR, "class_labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            classes = json.load(f)
    else:
        classes = CLASSES

    return model, classes


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess uploaded image for model input."""
    # Convert to RGB if grayscale or RGBA
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict(model, image: Image.Image, classes: list) -> dict:
    """Run prediction on an image."""
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)[0]

    results = {}
    for i, cls in enumerate(classes):
        results[cls] = float(predictions[i])

    return results


# ─── Streamlit App ────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Brain Tumor MRI Classifier",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
        }
        .result-box {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 1rem 0;
        }
        .confidence-bar {
            height: 24px;
            border-radius: 12px;
            margin: 4px 0;
        }
        .disclaimer {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1.5rem;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Brain Tumor MRI Classifier")
    st.markdown("**Upload a brain MRI scan to detect and classify tumors**")
    st.markdown("</div>", unsafe_allow_html=True)

    # Load model
    model, classes = load_model()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload MRI Brain Scan",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF",
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

        # Run prediction
        with st.spinner("Analyzing MRI scan..."):
            results = predict(model, image, classes)

        # Get top prediction
        top_class = max(results, key=results.get)
        top_confidence = results[top_class]
        top_label = CLASS_LABELS[top_class]

        st.markdown("---")

        # ─── Result ───────────────────────────────────────────────────────────
        if top_class == "notumor":
            st.success(f"### No Brain Tumor Detected")
            st.markdown(f"**Confidence: {top_confidence:.1%}**")
        else:
            st.error(f"### Brain Tumor Detected: {top_label}")
            st.markdown(f"**Confidence: {top_confidence:.1%}**")

        # ─── Confidence breakdown ─────────────────────────────────────────────
        st.markdown("#### Confidence Breakdown")

        # Sort by confidence descending
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        for cls, conf in sorted_results:
            label = CLASS_LABELS[cls]
            color = CLASS_COLORS[cls]
            bar_width = conf * 100

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"""<div style="margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; margin-bottom: 2px;">
                        <div style="width: {bar_width}%; background: {color}; height: 22px;
                             border-radius: 6px; display: flex; align-items: center;
                             padding-left: 8px; color: white; font-size: 12px; min-width: 40px;">
                            {conf:.1%}
                        </div>
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(f"**{label}**")

        # ─── Tumor Information ────────────────────────────────────────────────
        st.markdown("---")
        info = TUMOR_INFO[top_class]

        if top_class != "notumor":
            st.markdown(f"#### About {top_label}")
        else:
            st.markdown("#### Result Details")

        st.markdown(f"**Description:** {info['description']}")

        if top_class != "notumor":
            st.markdown(f"**Severity Level:** {info['severity']}")
            st.markdown(f"**Recommended Action:** {info['action']}")

        # ─── Disclaimer ───────────────────────────────────────────────────────
        st.markdown(
            """
            <div class="disclaimer">
                <strong>Medical Disclaimer:</strong> This tool is for educational and
                research purposes only. It should NOT be used as a substitute for
                professional medical diagnosis. Always consult a qualified medical
                professional for any health concerns.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        # Landing state
        st.markdown("---")
        st.markdown("### How It Works")
        st.markdown(
            """
            1. **Upload** a brain MRI scan image (JPG, PNG, etc.)
            2. **AI Analysis** — The EfficientNetB3 model analyzes the scan
            3. **Results** — Get instant classification with confidence scores

            **Supported Classifications:**
            | Class | Description |
            |-------|-------------|
            | Glioma Tumor | Tumor from glial cells — most common brain tumor |
            | Meningioma Tumor | Tumor from brain membranes — usually benign |
            | No Tumor | Normal brain scan — no abnormalities detected |
            | Pituitary Tumor | Tumor in pituitary gland — affects hormones |
            """
        )

        st.markdown("---")
        st.markdown(
            """
            <div class="disclaimer">
                <strong>Medical Disclaimer:</strong> This tool is for educational and
                research purposes only. It should NOT be used as a substitute for
                professional medical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
