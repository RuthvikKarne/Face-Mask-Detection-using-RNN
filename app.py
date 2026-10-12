import os
import pickle
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Ensure FaceMaskRNNModel class is available in scope for unpickling
from face_mask_rnn_model import FaceMaskRNNModel

# Streamlit Page Configuration
st.set_page_config(
    page_title="Face Mask Detection (RNN)",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling for modern look
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1d4ed8, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.15rem;
        margin-bottom: 1.5rem;
    }
    .upload-box {
        border: 2px dashed #3b82f6;
        border-radius: 12px;
        padding: 1rem;
        background-color: #f8fafc;
        margin-bottom: 1.5rem;
    }
    .status-card-mask {
        background: linear-gradient(135deg, #065f46, #059669);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.25);
        text-align: center;
        margin: 1rem 0;
    }
    .status-card-nomask {
        background: linear-gradient(135deg, #991b1b, #dc2626);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.25);
        text-align: center;
        margin: 1rem 0;
    }
    .result-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model(model_path="face_mask_rnn_model.pkl"):
    """Load serialized model from pickle file."""
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Header Section
st.markdown("""
<div class="main-header">
    <div class="main-title">😷 Face Mask Detection using RNN</div>
    <div class="sub-title">Upload any picture below — the Recurrent Neural Network will detect whether a face mask is present!</div>
</div>
""", unsafe_allow_html=True)

# Load the trained model
model = load_trained_model("face_mask_rnn_model.pkl")

if model is None:
    st.error("⚠️ Model file `face_mask_rnn_model.pkl` not found! Please run `py train_model.py` first.")
    st.stop()

# Helper function to display prediction for an image
def display_detection(image_obj, file_title="Uploaded Picture"):
    img_np = np.array(image_obj)
    if len(img_np.shape) == 3 and img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    t0 = time.time()
    pred = model.predict(img_np)
    latency_ms = (time.time() - t0) * 1000

    label = pred['label']
    confidence = pred['confidence'] * 100
    prob_mask = pred['probability_with_mask'] * 100
    prob_nomask = (1.0 - pred['probability_with_mask']) * 100

    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1, 1.8])

    with c1:
        st.markdown(f"**📷 {file_title}**")
        st.image(image_obj)

    with c2:
        st.markdown("**🔬 RNN Scanlines (32×32)**")
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
        gray_32 = cv2.resize(gray, (32, 32))
        st.image(gray_32, clamp=True, caption="Sequential Input")

    with c3:
        st.markdown("**🎯 Detection Result**")
        if label == "with_mask":
            st.markdown(f"""
            <div class="status-card-mask">
                <h2 style="margin:0; font-size: 1.8rem;">✅ MASK DETECTED</h2>
                <p style="margin:0.3rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                    <b>Status:</b> Wearing a Face Mask<br>
                    <b>Confidence:</b> {confidence:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card-nomask">
                <h2 style="margin:0; font-size: 1.8rem;">⚠️ NO MASK DETECTED</h2>
                <p style="margin:0.3rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                    <b>Status:</b> Not Wearing a Mask<br>
                    <b>Confidence:</b> {confidence:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("**Probability Meter:**")
        st.progress(float(pred['probability_with_mask']))

        p1, p2 = st.columns(2)
        with p1:
            st.metric("😷 P(With Mask)", f"{prob_mask:.1f}%")
        with p2:
            st.metric("👤 P(Without Mask)", f"{prob_nomask:.1f}%")

        st.caption(f"⚡ Inference Time: {latency_ms:.1f} ms | Model: `face_mask_rnn_model.pkl`")

    st.markdown('</div>', unsafe_allow_html=True)

# Main Method Selection Tabs
main_tabs = st.tabs(["📤 Upload Pictures", "🖼️ Sample Pictures", "📊 Model Performance"])

# ----------------- TAB 1: UPLOAD PICTURES (MAIN USER FEATURE) -----------------
with main_tabs[0]:
    st.subheader("📤 Upload Pictures for Detection")
    st.write("Upload one or more face photos (.jpg, .jpeg, .png, .webp). The RNN model will analyze the sequential facial scanlines and detect masks instantly.")

    uploaded_files = st.file_uploader(
        "Choose face photo(s)...",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Drag and drop or browse images from your device"
    )

    if uploaded_files:
        st.success(f"Processing {len(uploaded_files)} uploaded picture(s)...")
        for idx, file in enumerate(uploaded_files):
            try:
                pil_img = Image.open(file)
                display_detection(pil_img, file_title=f"Picture #{idx+1}: {file.name}")
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")
    else:
        st.info("💡 **Drag and drop any picture above, or click 'Browse files' to upload!**")

# ----------------- TAB 2: SAMPLE PICTURES -----------------
with main_tabs[1]:
    st.subheader("🖼️ Test with Pre-Loaded Sample Pictures")
    st.write("Want to test without uploading your own photo? Choose from these sample images from the dataset:")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.markdown("### 😷 With Mask Samples")
        wm_dir = os.path.join("sample_images", "with_mask")
        if os.path.isdir(wm_dir):
            wm_samples = [f for f in os.listdir(wm_dir) if f.lower().endswith(('.jpg', '.png'))][:6]
            chosen_wm = st.selectbox("Select 'With Mask' sample:", wm_samples, key="sample_wm")
            if st.button("Detect on this 'With Mask' picture", key="btn_wm"):
                sample_img = Image.open(os.path.join(wm_dir, chosen_wm))
                display_detection(sample_img, file_title=f"Sample: {chosen_wm}")
        else:
            st.info("Sample images not available. Please upload your own image in the Upload tab.")

    with col_btn2:
        st.markdown("### 👤 Without Mask Samples")
        wom_dir = os.path.join("sample_images", "without_mask")
        if os.path.isdir(wom_dir):
            wom_samples = [f for f in os.listdir(wom_dir) if f.lower().endswith(('.jpg', '.png'))][:6]
            chosen_wom = st.selectbox("Select 'Without Mask' sample:", wom_samples, key="sample_wom")
            if st.button("Detect on this 'Without Mask' picture", key="btn_wom"):
                sample_img = Image.open(os.path.join(wom_dir, chosen_wom))
                display_detection(sample_img, file_title=f"Sample: {chosen_wom}")
        else:
            st.info("Sample images not available. Please upload your own image in the Upload tab.")

# ----------------- TAB 3: PERFORMANCE -----------------
with main_tabs[2]:
    st.subheader("📊 Model Performance & Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", "92.60%")
    m2.metric("Test Precision", "90.30%")
    m3.metric("Test Recall", "95.65%")
    m4.metric("Test F1-Score", "92.90%")

    st.markdown("---")
    perf_col1, perf_col2 = st.columns([1.5, 1])
    with perf_col1:
        st.markdown("**Training Loss & Accuracy Curves:**")
        if os.path.exists("training_curves.png"):
            st.image("training_curves.png", caption="Adam Optimizer Training Progression")
    with perf_col2:
        st.markdown("**Confusion Matrix (500 Test Samples):**")
        st.table({
            "Predicted: No Mask": [221, 11],
            "Predicted: Mask": [26, 242]
        })
        st.caption("Rows: [Actual Without Mask, Actual With Mask]")

