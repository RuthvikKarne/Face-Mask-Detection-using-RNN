# Face-Mask-Detection-using-RNN

An end-to-end Deep Learning pipeline and interactive web application for **Face Mask Detection** using a custom **Recurrent Neural Network (RNN)** with **Spatial-Sequential Lower-Region Attention Pooling**, serialized with **Pickle** (`.pkl`), and deployed with **Streamlit**.

---

## 🚀 Key Highlights

- **🧠 Spatial-Sequential Scanline RNN**: Models 2D facial images row-by-row as an ordered sequence of horizontal scanlines (32 timesteps $\times$ 96 RGB color features).
- **🎯 Dual-Region Attention Pooling**: Combines global sequence pooling with targeted lower-region pooling (Rows 14–31) to capture mask presence without vanishing memory at the chin.
- **⚡ Custom Backpropagation Through Time (BPTT)**: Fully vectorized gradient computation with Adam optimizer.
- **💾 Pickle Model Serialization**: The trained model is saved to `face_mask_rnn_model.pkl` for one-line import and instant inference.
- **🌐 Streamlit Web Application**: Modern user interface supporting drag-and-drop picture uploads, probability meters, and scanline visualizations.
- **📓 Complete Jupyter Notebook**: Detailed educational guide in `face_mask_detection_rnn.ipynb`.

---

## 📊 Performance Metrics

Evaluated on 500 unseen real-world test images:

| Metric | Score |
| :--- | :---: |
| **Mask Detection Recall** | **95.65%** (242 / 253 masks detected) |
| **Overall Accuracy** | **92.60%** |
| **Precision** | **90.30%** |
| **F1-Score** | **92.90%** |

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RuthvikKarne/Face-Mask-Detection-using-RNN.git
   cd Face-Mask-Detection-using-RNN
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the Application

### 1. Launch the Streamlit Web App
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. Drag and drop any face picture to detect mask status in real time!

### 2. Run Standalone CLI Inference
```bash
python predict_example.py
```

### 3. Retrain the RNN Model
```bash
python train_model.py
```

### 4. Interactive Jupyter Notebook
Open `face_mask_detection_rnn.ipynb` in VS Code or JupyterLab and run all cells.

---

## 📦 How to Import the Model with Pickle

```python
import pickle

# Load the trained model
with open("face_mask_rnn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Run prediction on any image path or numpy array
result = model.predict("path/to/image.jpg")

print("Status:    ", result["label"])               # 'with_mask' or 'without_mask'
print("Confidence:", f"{result['confidence'] * 100:.2f}%")
print("P(Mask):   ", f"{result['probability_with_mask'] * 100:.2f}%")
```

---

## 📁 Repository Structure

```text
├── app.py                         # Streamlit web application
├── face_mask_rnn_model.py         # FaceMaskRNNModel class & inference engine
├── face_mask_rnn_model.pkl        # Serialized trained model weights (Pickle)
├── train_model.py                 # Training script with Adam optimizer
├── predict_example.py             # CLI inference demonstration
├── face_mask_detection_rnn.ipynb  # Complete Jupyter Notebook pipeline
├── requirements.txt               # Python package dependencies
├── training_curves.png            # Loss and accuracy progression plot
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

---

## 📜 License
This project is open-source and available under the MIT License.
