# HackRush 2026: Custom Loan Default Predictor & Risk Analytics

An end-to-end Machine Learning solution designed to predict retail loan default risk and provide decision explainability using LightGBM and SHAP. 

This repository was cloned from the boilerplate and configured to point to your personal repository: **[HackRush_2026_IIT-GN](https://github.com/ayush300302/HackRush_2026_IIT-GN)**.

---

## 🧩 Workspace Structure

The project has been organized to keep development and serving clean and modular:

```
HackRush_2026/
│
├── HackRush_Hackhon_IITGN/    # Your Custom Workspace (From Scratch)
│   ├── notebooks/             # Exploratory analysis & feature profiling
│   │   └── eda_and_profiling.ipynb
│   │
│   ├── src/                   # Custom ML pipeline modules
│   │   ├── data_loader.py     # Downcast types, chunked data loader
│   │   ├── preprocessor.py    # Imputation, encoding, winsorization
│   │   └── model.py           # Training, splitting, evaluation metrics
│   │
│   ├── train.py               # Training pipeline execution script
│   ├── app.py                 # Streamlit UI dashboard
│   └── requirements.txt       # Project dependencies
│
├── Retail_Loan_Default_Prediction/ # Reference codebase (Jupyter research files)
└── SmartLend_App/             # Reference boilerplate application
```

---

## 🚀 How to Run the Custom Project

### 1. Setup Environment & Dependencies
Initialize your virtual environment (if not already done) and install the libraries:
```bash
pip install -r HackRush_Hackhon_IITGN/requirements.txt
```

### 2. Run Exploratory Data Analysis
You can open and execute the custom notebook in `HackRush_Hackhon_IITGN/notebooks/eda_and_profiling.ipynb` to analyze feature correlations and risk distributions on a 100k sample.

### 3. Execute Model Training
To train the LightGBM classifier on the full 2.26 million historical loans dataset:
```bash
cd HackRush_Hackhon_IITGN
python train.py
```
*This will run memory-efficient loading, clean the dataset, train the LightGBM model with validation early stopping, extract frequency mappings, and save the serialized weights/mappings to `models/model_assets.joblib`.*

### 4. Launch the Streamlit Dashboard
To run the local web dashboard for interactive scoring:
```bash
streamlit run app.py
```
*Go to **http://localhost:8501** in your browser to test inputs in Rupees, check Expected Loss (EL), and view local decision explanations.*
