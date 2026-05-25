# SmartLend: Simplified & Modular Loan Default Predictor

SmartLend is an interactive, modular Machine Learning application designed to predict and explain retail loan default risk. This repository is built **from scratch** to be clean, heavily commented, and extremely easy to understand for developers and judges.

---

## 🧩 Folder Structure
Every module is dedicated to a single, distinct task in the data science pipeline:

```
HackRush_Hackhon_IITGN/
│
├── src/                       # Modular source code
│   ├── data_loader.py         # Memory-efficient raw data loading
│   ├── preprocessor.py        # Data cleaning, outlier clipping, and frequency mapping
│   └── model.py               # Model training, validation, and evaluation routines
│
├── models/                    # Serialized ML assets
│   └── model_assets.joblib    # Save file containing model weights, feature mappings, and medians
│
├── train.py                   # Runner script to load data, train model, and save assets
├── app.py                     # Streamlit web dashboard for credit scoring & SHAP explanations
├── requirements.txt           # Python library dependencies
└── README.md                  # System overview and guide
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
Initialize and activate your virtual environment, then install the packages:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Train the LightGBM classifier and extract the categorical mapping dictionaries:
```bash
python train.py
```
*This script will load the raw CSV, preprocess the features, train the model, and output the serialized file `models/model_assets.joblib`.*

### 3. Launch the Streamlit Dashboard
Start the local interactive credit portal:
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser to:
1. Score loan applications in Indian Rupees (₹).
2. Examine the dynamic decision (Approved vs. Rejected) and Expected Loss.
3. Check the interactive SHAP chart explaining exactly why the decision was made.
