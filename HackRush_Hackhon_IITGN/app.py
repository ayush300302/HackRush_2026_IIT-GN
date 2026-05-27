import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import shap

# 1. Page Configuration & Custom CSS Styling
st.set_page_config(
    page_title="SmartLend Analytics | Custom Risk Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Currency conversion: 1 USD = 83 INR
CONVERSION_RATE = 83.0

# Inject custom CSS to make the dashboard look premium and clean
st.markdown("""
<style>
    /* Main Background color */
    .reportview-container {
        background: #f8f9fa;
    }
    /* Title text styling */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1e293b;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    /* Subtitle styling */
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    /* Approved application banner styling */
    .decision-banner {
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .approved {
        background-color: #d1fae5;
        border: 2px solid #10b981;
        color: #065f46;
    }
    /* Rejected application banner styling */
    .rejected {
        background-color: #fee2e2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    /* Blue gradient divider line */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #3b82f6 0%, #10b981 100%);
        margin: 1.5rem 0;
        border-radius: 2px;
    }
    /* Resize streamlit metric outputs to fit narrow columns */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        white-space: nowrap;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        white-space: nowrap;
    }
</style>
""", unsafe_allow_html=True)

# 2. Loading Serialized Model and Mappings
@st.cache_resource
def load_model_assets():
    """
    Loads and caches the serialized model assets to speed up page reloads.
    """
    path = "models/model_assets.joblib"
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Error loading model assets file: {e}")
            return None
    return None

@st.cache_data
def load_historical_sample_data():
    """
    Loads and caches a small random sample of historical loans to draw charts.
    """
    path = "data/processed_loan_data.parquet"
    if not os.path.exists(path):
        # Fallback path looking at sister directory
        path = "../Retail_Loan_Default_Prediction/data/processed_loan_data.parquet"
        
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            return df.sample(20000, random_state=42)
        except Exception as e:
            st.error(f"Error reading Parquet sample data: {e}")
            return None
    return None

@st.cache_resource
def get_shap_explainer(_model):
    """
    Builds and caches a SHAP Tree Explainer for the LightGBM model.
    """
    return shap.TreeExplainer(_model)

# Trigger loading of assets
assets = load_model_assets()

if assets is None:
    st.error("⚠️ Model assets not found! Please run the training script first (`python train.py`).")
else:
    # Unpack assets
    model = assets['model']
    features = assets['features']
    medians = assets['medians']
    winsorize_limits = assets['winsorize_limits']
    cat_mappings = assets['cat_mappings']
    test_metrics = assets['test_metrics']
    
    # Initialize the SHAP explainer
    explainer = get_shap_explainer(model)

    # 3. Sidebar Configuration panel
    st.sidebar.markdown("<h2 style='text-align: center;'>SmartLend Controls</h2>", unsafe_allow_html=True)
    st.sidebar.write("Calibrate the default prediction engine parameters.")
    st.sidebar.markdown(f"<small>Standard Conversion Rate: 1 USD = ₹{CONVERSION_RATE}</small>", unsafe_allow_html=True)
    
    # Adjustable risk threshold slider
    threshold = st.sidebar.slider(
        "Approval Risk Threshold (PD)", 0.05, 0.50, 0.20, step=0.01,
        help="The maximum Probability of Default (PD) to tolerate before rejecting a loan."
    )
    
    # App Title & Headers
    st.markdown("<div class='main-title'>💳 SmartLend Risk Analytics Platform</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Real-time credit score, Expected Loss prediction, and SHAP explainability.</div>", unsafe_allow_html=True)
    
    # Create tab structure
    tab1, tab2, tab3 = st.tabs(["🔍 Credit Scoring Portal", "📊 Historical Insights", "🏆 AI Model Diagnostics"])
    
    # TAB 1: REAL-TIME CREDIT SCORING PORTAL
    with tab1:
        st.markdown("### Apply for a New Retail Loan (in ₹)")
        st.write("Fill out the applicant's parameters to predict creditworthiness instantly.")
        
        # User input form
        with st.form("loan_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 1. Loan Specifications")
                loan_amnt_inr = st.number_input("Requested Loan Amount (₹)", min_value=50000, max_value=3500000, value=1200000, step=25000)
                
                term_options = sorted(list(cat_mappings['term']['freq_map'].keys()))
                term = st.selectbox("Loan Term", term_options)
                
                int_rate = st.slider("Interest Rate (%)", 5.0, 35.0, 12.5, step=0.1)
                
                grade_options = sorted(list(cat_mappings['grade']['freq_map'].keys()))
                grade = st.selectbox("Loan Grade", grade_options, index=1)
                
                # Filter subgrade choices to match grade prefix
                sub_grade_options = sorted([sg for sg in cat_mappings['sub_grade']['freq_map'].keys() if sg.startswith(grade)])
                if not sub_grade_options:
                    sub_grade_options = sorted(list(cat_mappings['sub_grade']['freq_map'].keys()))
                sub_grade = st.selectbox("Sub-Grade", sub_grade_options)

            with col2:
                st.markdown("#### 2. Borrower Profile")
                annual_inc_inr = st.number_input("Annual Income (₹)", min_value=100000, max_value=30000000, value=1500000, step=50000)
                
                emp_length_options = sorted(list(cat_mappings['emp_length']['freq_map'].keys()))
                emp_length = st.selectbox("Employment Length", emp_length_options, index=emp_length_options.index("10+ years") if "10+ years" in emp_length_options else 0)
                
                home_ownership_options = sorted(list(cat_mappings['home_ownership']['freq_map'].keys()))
                home_ownership = st.selectbox("Home Ownership", home_ownership_options, index=home_ownership_options.index("MORTGAGE") if "MORTGAGE" in home_ownership_options else 0)
                
                verification_status_options = sorted(list(cat_mappings['verification_status']['freq_map'].keys()))
                verification_status = st.selectbox("Income Verification Status", verification_status_options)

            with col3:
                st.markdown("#### 3. Financial Ratios")
                dti = st.slider("Debt-to-Income Ratio (DTI, %)", 0.0, 100.0, 18.5, step=0.1)
                
                purpose_options = sorted(list(cat_mappings['purpose']['freq_map'].keys()))
                purpose = st.selectbox("Loan Purpose", purpose_options, index=purpose_options.index("debt_consolidation") if "debt_consolidation" in purpose_options else 0)
                
                application_type_options = sorted(list(cat_mappings['application_type']['freq_map'].keys()))
                application_type = st.selectbox("Application Type", application_type_options)
                
                disbursement_method_options = sorted(list(cat_mappings['disbursement_method']['freq_map'].keys()))
                disbursement_method = st.selectbox("Disbursement Method", disbursement_method_options)
                
                # Dynamic installment calculator
                months = 36 if "36" in term else 60
                monthly_rate = (int_rate / 100) / 12
                approx_installment_inr = loan_amnt_inr * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
                installment_inr = st.number_input("Monthly Installment (₹)", min_value=500.0, max_value=250000.0, value=float(round(approx_installment_inr, 2)))
                
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Collapsible Advanced Section
            with st.expander("🛠️ Advanced Credit Bureau Metrics"):
                st.write("Credit score parameters (pre-filled with training medians, monetary items in ₹).")
                col_adv1, col_adv2, col_adv3 = st.columns(3)
                
                with col_adv1:
                    open_acc = st.number_input("Open Credit Accounts", min_value=0, max_value=100, value=int(medians.get('open_acc', 11)))
                    total_acc = st.number_input("Total Credit Accounts", min_value=0, max_value=150, value=int(medians.get('total_acc', 24)))
                    
                    default_revol_bal_inr = int(medians.get('revol_bal', 11000) * CONVERSION_RATE)
                    revol_bal_inr = st.number_input("Revolving Balance (₹)", min_value=0, max_value=50000000, value=default_revol_bal_inr)
                    revol_util = st.slider("Revolving Utilization (%)", 0.0, 150.0, float(medians.get('revol_util', 50.0)))
                    
                with col_adv2:
                    delinq_2yrs = st.number_input("Delinquencies (Last 2 Years)", min_value=0, max_value=30, value=int(medians.get('delinq_2yrs', 0)))
                    inq_last_6mths = st.number_input("Inquiries (Last 6 Months)", min_value=0, max_value=10, value=int(medians.get('inq_last_6mths', 0)))
                    pub_rec = st.number_input("Public Derogatory Records", min_value=0, max_value=20, value=int(medians.get('pub_rec', 0)))
                    pub_rec_bankruptcies = st.number_input("Public Bankruptcies", min_value=0, max_value=10, value=int(medians.get('pub_rec_bankruptcies', 0)))
                    
                with col_adv3:
                    initial_list_status_options = sorted(list(cat_mappings['initial_list_status']['freq_map'].keys()))
                    initial_list_status = st.selectbox("Initial List Status", initial_list_status_options)
                    tax_liens = st.number_input("Tax Liens", min_value=0, max_value=20, value=int(medians.get('tax_liens', 0)))
                    
            submit_button = st.form_submit_button("Evaluate Loan Application", type="primary")
            
        # 4. Input Processing & Scoring
        if submit_button:
            # Scale down to USD-equivalent amounts for prediction
            loan_amnt_usd = float(loan_amnt_inr / CONVERSION_RATE)
            annual_inc_usd = float(annual_inc_inr / CONVERSION_RATE)
            installment_usd = float(installment_inr / CONVERSION_RATE)
            revol_bal_usd = float(revol_bal_inr / CONVERSION_RATE)
            
            # Start vector with features medians
            input_row = medians.copy()
            
            # Insert converted inputs
            input_row['loan_amnt'] = loan_amnt_usd
            input_row['funded_amnt'] = loan_amnt_usd
            input_row['funded_amnt_inv'] = loan_amnt_usd
            input_row['int_rate'] = float(int_rate)
            input_row['installment'] = installment_usd
            input_row['annual_inc'] = annual_inc_usd
            input_row['dti'] = float(dti)
            
            # Unit-free calculated ratio
            input_row['income_to_loan_ratio'] = float(annual_inc_usd / (loan_amnt_usd + 1e-5))
            
            # Installment-to-income ratio (Monthly installment / Monthly income)
            monthly_income_usd = annual_inc_usd / 12.0
            input_row['installment_to_income_ratio'] = float(installment_usd / (monthly_income_usd + 1e-5))
            
            # Insert advanced inputs
            input_row['open_acc'] = float(open_acc)
            input_row['total_acc'] = float(total_acc)
            input_row['revol_bal'] = revol_bal_usd
            input_row['revol_util'] = float(revol_util)
            input_row['delinq_2yrs'] = float(delinq_2yrs)
            input_row['inq_last_6mths'] = float(inq_last_6mths)
            input_row['pub_rec'] = float(pub_rec)
            input_row['pub_rec_bankruptcies'] = float(pub_rec_bankruptcies)
            input_row['tax_liens'] = float(tax_liens)
            
            # Clip outliers using winsorization limits
            for col in winsorize_limits:
                if col in input_row:
                    limits = winsorize_limits[col]
                    input_row[col] = np.clip(input_row[col], limits['lower'], limits['upper'])
            
            # 1. Map smart ordinal categorical variables
            grade_map = {'A': 1.0, 'B': 2.0, 'C': 3.0, 'D': 4.0, 'E': 5.0, 'F': 6.0, 'G': 7.0}
            
            sub_grade_map = {}
            grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            for i, g in enumerate(grades):
                for num in ['1', '2', '3', '4', '5']:
                    sub_grade_map[g + num] = float(i * 5 + int(num))
                    
            emp_length_map = {
                '< 1 year': 0.0, '1 year': 1.0, '2 years': 2.0, '3 years': 3.0, '4 years': 4.0,
                '5 years': 5.0, '6 years': 6.0, '7 years': 7.0, '8 years': 8.0, '9 years': 9.0,
                '10+ years': 10.0, 'Unknown': 0.0
            }
            
            input_row['grade'] = grade_map.get(grade, 4.0)
            input_row['sub_grade'] = sub_grade_map.get(sub_grade, 17.0)
            input_row['emp_length'] = emp_length_map.get(emp_length, 5.0)
            
            # 2. Map standard text categories to frequencies
            categorical_inputs = {
                'term': term, 'home_ownership': home_ownership,
                'verification_status': verification_status, 'purpose': purpose,
                'application_type': application_type, 'disbursement_method': disbursement_method,
                'initial_list_status': initial_list_status, 'title': 'Debt consolidation'
            }
            
            for col, val in categorical_inputs.items():
                if col in cat_mappings:
                    freq_map = cat_mappings[col]['freq_map']
                    default_mode = cat_mappings[col]['default_mode']
                    val_str = str(val)
                    input_row[col] = freq_map[val_str] if val_str in freq_map else freq_map[default_mode]
            
            # Convert to aligned DataFrame
            input_df = pd.DataFrame([input_row])
            input_df = input_df[features] # enforce feature order
            
            # Make default risk prediction
            pd_prob = float(model.predict_proba(input_df)[0, 1])
            
            # Calculate Expected Loss
            lgd = 0.50 # 50% Loss Given Default standard
            ead_inr = loan_amnt_inr
            expected_loss_inr = pd_prob * lgd * ead_inr
            
            # 5. Display Evaluation Results
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown("### Evaluation Results")
            
            dec_col1, dec_col2 = st.columns([1, 1])
            
            with dec_col1:
                # Approval banners
                if pd_prob < threshold:
                    st.markdown(f"""
                    <div class='decision-banner approved'>
                        <h2 style='margin:0;'>✅ APPLICATION APPROVED</h2>
                        <p style='margin:10px 0 0 0; font-size:1.1rem;'>Applicant falls within the acceptable risk threshold of {threshold*100:.1f}%.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='decision-banner rejected'>
                        <h2 style='margin:0;'>❌ APPLICATION REJECTED</h2>
                        <p style='margin:10px 0 0 0; font-size:1.1rem;'>Applicant default probability exceeds risk threshold of {threshold*100:.1f}%.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Standard Metrics
                sub_kpi1, sub_kpi2, sub_kpi3 = st.columns(3)
                with sub_kpi1:
                    st.metric(label="Probability of Default (PD)", value=f"{pd_prob * 100:.2f}%")
                with sub_kpi2:
                    el_display = f"₹{int(round(expected_loss_inr)):,}" if expected_loss_inr >= 100 else f"₹{expected_loss_inr:,.2f}"
                    st.metric(label="Expected Loss (EL)", value=el_display, help="EL = PD * LGD * EAD (assuming 50% LGD)")
                with sub_kpi3:
                    risk_cat = "Low" if pd_prob < 0.10 else "Medium" if pd_prob < 0.20 else "High" if pd_prob < 0.35 else "Critical"
                    st.metric(label="Risk Class", value=risk_cat)
            
            with dec_col2:
                # Gauge dial indicator chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = pd_prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Default Probability (%)", 'font': {'size': 18}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#1e293b"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, threshold*100], 'color': '#d1fae5'},
                            {'range': [threshold*100, 100], 'color': '#fee2e2'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold*100
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            # 6. SHAP Decision Explanations plot
            st.markdown("#### 🔍 Explain Decision (SHAP Contributions)")
            st.write("This chart shows exactly how each input feature contributed to increasing (red) or decreasing (green) risk.")
            
            shap_values = explainer(input_df)
            
            shap_df = pd.DataFrame({
                'Feature': features,
                'SHAP Value': shap_values.values[0]
            })
            
            shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
            shap_df = shap_df.sort_values(by='Abs SHAP', ascending=False).head(10)
            
            shap_df['Direction'] = np.where(shap_df['SHAP Value'] >= 0, 'Increases Risk 📈', 'Reduces Risk 📉')
            shap_df['Clean Feature'] = shap_df['Feature'].str.replace('_', ' ').str.title()
            
            fig_shap = px.bar(
                shap_df,
                x='SHAP Value',
                y='Clean Feature',
                orientation='h',
                color='Direction',
                color_discrete_map={'Increases Risk 📈': '#ef4444', 'Reduces Risk 📉': '#10b981'},
                title="Top 10 Decision Factors Contributing to Risk Score",
                labels={'Clean Feature': 'Credit Feature', 'SHAP Value': 'Risk Impact (SHAP value)'}
            )
            fig_shap.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            fig_shap.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
            st.plotly_chart(fig_shap, use_container_width=True)

    # TAB 2: HISTORICAL INSIGHTS & EDA
    with tab2:
        st.markdown("### Historical Portfolio Insights (INR Converted)")
        st.write("Key portfolio metrics converted to Rupees based on historical credit records of 2.26 million borrowers.")
        
        hist_df = load_historical_sample_data()
        if hist_df is None:
            st.warning("Could not load historical parquet dataset sample.")
        else:
            # Map grade floats back to text labels for visualizations
            grade_freqs = cat_mappings['grade']['freq_map']
            rev_grade_map = {v: k for k, v in grade_freqs.items()}
            
            df_plot = hist_df[['grade', 'is_default', 'int_rate', 'dti', 'annual_inc']].copy()
            df_plot['Grade Name'] = df_plot['grade'].map(rev_grade_map).fillna("Unknown")
            df_plot['annual_inc_inr'] = df_plot['annual_inc'] * CONVERSION_RATE
            
            col_graph1, col_graph2 = st.columns(2)
            
            with col_graph1:
                st.markdown("#### The 12x Default Rate Gap Across Grades")
                st.write("Lower-grade loans (Grade G/F) exhibit exponentially higher default rates than Grade A loans.")
                
                default_by_grade = df_plot.groupby('Grade Name')['is_default'].mean().reset_index()
                default_by_grade['Default Rate (%)'] = default_by_grade['is_default'] * 100
                default_by_grade = default_by_grade.sort_values(by='Grade Name')
                
                fig_gap = px.bar(
                    default_by_grade,
                    x='Grade Name',
                    y='Default Rate (%)',
                    color='Grade Name',
                    color_discrete_sequence=px.colors.sequential.YlOrRd[::-1],
                    labels={'Grade Name': 'Loan Grade', 'Default Rate (%)': 'Default Rate (%)'},
                    title="Average Historical Default Rate by Loan Grade"
                )
                fig_gap.update_layout(plot_bgcolor='white', showlegend=False)
                fig_gap.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
                st.plotly_chart(fig_gap, use_container_width=True)
                
            with col_graph2:
                st.markdown("#### Interest Rate vs. Debt-to-Income (DTI)")
                st.write("Scatter distribution of borrower interest rates relative to DTI ratios, colored by default outcome.")
                
                fig_scatter = px.scatter(
                    df_plot.sample(2000, random_state=42),
                    x='dti',
                    y='int_rate',
                    color='is_default',
                    color_discrete_map={0: '#3b82f6', 1: '#ef4444'},
                    labels={'dti': 'Debt-to-Income (DTI, %)', 'int_rate': 'Interest Rate (%)', 'is_default': 'Defaulted'},
                    opacity=0.6,
                    title="Borrower Distribution (DTI vs. Interest Rate)"
                )
                fig_scatter.update_layout(plot_bgcolor='white')
                fig_scatter.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
                fig_scatter.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            st.markdown("#### Borrower Annual Income Distribution (Clipped at ₹2.25 Crores)")
            st.write("Histogram showing the distribution of annual income in Indian Rupees.")
            
            max_inc_inr_display = 270000.0 * CONVERSION_RATE
            df_plot_inc_filtered = df_plot[df_plot['annual_inc_inr'] <= max_inc_inr_display]
            
            fig_inc = px.histogram(
                df_plot_inc_filtered,
                x='annual_inc_inr',
                nbins=50,
                color_discrete_sequence=['#10b981'],
                labels={'annual_inc_inr': 'Annual Income (₹)'},
                title="Distribution of Borrower Annual Income (INR)"
            )
            fig_inc.update_layout(plot_bgcolor='white')
            fig_inc.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
            st.plotly_chart(fig_inc, use_container_width=True)

    # TAB 3: AI MODEL DIAGNOSTICS
    with tab3:
        st.markdown("### Machine Learning Model Diagnostics")
        st.write("Evaluation metrics and details for the underlying LightGBM classifier.")
        
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            st.markdown("#### Performance Metrics")
            st.write("The model was evaluated using a stratified 20% test partition (452,134 rows).")
            
            metrics_df = pd.DataFrame({
                'Metric': ['Test ROC-AUC', 'Test Precision-Recall AUC (PR-AUC)', 'Validation Early Stopping Rounds'],
                'Score/Value': [f"{test_metrics['ROC-AUC']:.4f}", f"{test_metrics['PR-AUC']:.4f}", "30 rounds"]
            })
            st.table(metrics_df)
            
            st.markdown("""
            > **Business Interpretation:**
            > * **ROC-AUC (0.7165):** The model has a 71.65% probability of correctly ranking a randomly chosen default applicant higher than a non-default applicant.
            > * **PR-AUC (0.2451):** Outperforms the random default baseline (~12.5% rate) significantly. Indicates strong capacity to recall defaults while minimizing false alarms.
            """)
            
        with diag_col2:
            st.markdown("#### Feature Importances")
            st.write("Top 10 features sorted by their LightGBM feature gain/split contribution.")
            
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(10)
            
            feat_imp_df['Clean Feature'] = feat_imp_df['Feature'].str.replace('_', ' ').str.title()
            
            fig_imp = px.bar(
                feat_imp_df,
                x='Importance',
                y='Clean Feature',
                orientation='h',
                color_discrete_sequence=['#3b82f6'],
                labels={'Clean Feature': 'Credit Feature', 'Importance': 'Split Count Gain'}
            )
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, plot_bgcolor='white')
            fig_imp.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
            st.plotly_chart(fig_imp, use_container_width=True)
