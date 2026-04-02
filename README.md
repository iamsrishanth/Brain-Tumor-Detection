# Brain Tumor Detection

A deep-learning project for classifying brain MRI scans into four classes:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The repository includes:

- `train.py` — model training script based on **EfficientNetB3** (transfer learning + fine-tuning)
- `app.py` — **Streamlit** web app for uploading MRI images and getting predictions
- `Training/` and `Testing/` — image datasets organized by class
- `models/` — saved model artifacts (created after training)

---

## Project Structure

```text
Brain-Tumor-Detection/
├── app.py
├── train.py
├── requirements.txt
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── models/
```

---

## Setup

### 1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run default training:

```bash
python train.py
```

Run with custom hyperparameters:

```bash
python train.py --epochs 30 --fine_tune_epochs 20 --batch_size 16
```

During training, the best model is saved to:

- `models/best_model.keras`

At completion, final artifacts are written to:

- `models/brain_tumor_classifier.keras`
- `models/class_labels.json`

---

## Run the Web App

After training completes, start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal (typically `http://localhost:8501`) and upload an MRI image.

---

## Notes

- This project is intended for educational/research use and is **not a clinical diagnostic tool**.
- Model quality depends on data quality, preprocessing, and training configuration.

---

## License

No license file is currently included in this repository.
