import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import shap

# Set page config
st.set_page_config(
    page_title="SmartLend Analytics | Retail Loan Risk Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Conversion rate: 1 USD = 83 INR
CONVERSION_RATE = 83.0

# Custom premium CSS for styling
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #f8f9fa;
    }
    
    /* Header styling */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1e293b;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card container styling */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    /* Decisions banners */
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
    .rejected {
        background-color: #fee2e2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    
    /* Custom divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #3b82f6 0%, #10b981 100%);
        margin: 1.5rem 0;
        border-radius: 2px;
    }
    
    /* Adjust metric sizes to prevent truncation */
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

# Load model assets
@st.cache_resource
def load_assets():
    path = "models/model_assets.joblib"
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Error loading model assets: {e}")
            return None
    return None

# Load small data sample for historical charts
@st.cache_data
def load_historical_sample():
    path = "data/processed_loan_data.parquet"
    if os.path.exists(path):
        try:
            # We only need a subset for display to keep memory footprint low
            df = pd.read_parquet(path)
            return df.sample(20000, random_state=42)
        except Exception as e:
            st.error(f"Error loading historical sample: {e}")
            return None
    return None

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

assets = load_assets()

if assets is None:
    st.error("⚠️ Model assets not found! Please run the training script first (`train_and_serialize.py`).")
else:
    model = assets['model']
    features = assets['features']
    medians = assets['medians']
    winsorize_limits = assets['winsorize_limits']
    cat_mappings = assets['cat_mappings']
    test_metrics = assets['test_metrics']
    
    # Explainer for SHAP
    explainer = get_shap_explainer(model)

    # Sidebar Header
    st.sidebar.markdown("<h2 style='text-align: center;'>SmartLend Panel</h2>", unsafe_allow_html=True)
    st.sidebar.write("Configure risk engine parameters below.")
    st.sidebar.markdown(f"<small>Using Currency Conversion: 1 USD = ₹{CONVERSION_RATE}</small>", unsafe_allow_html=True)
    
    # Adjustable risk threshold
    threshold = st.sidebar.slider("Approval Risk Threshold (PD)", 0.05, 0.50, 0.20, step=0.01,
                                  help="The maximum Probability of Default (PD) to allow for loan approval.")
    
    # Title
    st.markdown("<div class='main-title'>💳 SmartLend Risk Analytics Platform</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Real-time loan credit scoring, default risk assessment, and decision explainability (INR Model).</div>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Credit Scoring Portal", "📊 Historical Insights", "🏆 AI Model Diagnostics"])
    
    # TAB 1: CREDIT SCORING
    with tab1:
        st.markdown("### Apply for a New Loan (in ₹)")
        st.write("Fill out the loan application parameters in Rupees to evaluate default probability in real-time.")
        
        # Form
        with st.form("loan_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 1. Loan Specifications")
                # Max value in dataset is $40,000 -> ₹33.2 Lakhs. We round to ₹35 Lakhs.
                loan_amnt_inr = st.number_input("Requested Loan Amount (₹)", min_value=50000, max_value=3500000, value=1200000, step=25000)
                
                # Fetch dynamically from mapping keys
                term_options = sorted(list(cat_mappings['term']['freq_map'].keys()))
                term = st.selectbox("Loan Term", term_options)
                
                int_rate = st.slider("Interest Rate (%)", 5.0, 35.0, 12.5, step=0.1)
                
                grade_options = sorted(list(cat_mappings['grade']['freq_map'].keys()))
                grade = st.selectbox("Loan Grade", grade_options, index=1)
                
                sub_grade_options = sorted([sg for sg in cat_mappings['sub_grade']['freq_map'].keys() if sg.startswith(grade)])
                if not sub_grade_options:
                    sub_grade_options = sorted(list(cat_mappings['sub_grade']['freq_map'].keys()))
                sub_grade = st.selectbox("Sub-Grade", sub_grade_options)

            with col2:
                st.markdown("#### 2. Borrower Profile")
                # Scale typical salaries in India (₹3 Lakhs to ₹3 Crores)
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
                
                # Approximate installment calculation
                months = 36 if "36" in term else 60
                monthly_rate = (int_rate / 100) / 12
                approx_installment_inr = loan_amnt_inr * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
                installment_inr = st.number_input("Monthly Installment (₹)", min_value=500.0, max_value=250000.0, value=float(round(approx_installment_inr, 2)))
                
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Collapsible Advanced Section
            with st.expander("🛠️ Advanced Credit Bureau Metrics"):
                st.write("Modify historical credit score parameters (pre-filled with training medians, monetary items in ₹).")
                col_adv1, col_adv2, col_adv3 = st.columns(3)
                
                with col_adv1:
                    open_acc = st.number_input("Open Credit Accounts", min_value=0, max_value=100, value=int(medians.get('open_acc', 11)))
                    total_acc = st.number_input("Total Credit Accounts", min_value=0, max_value=150, value=int(medians.get('total_acc', 24)))
                    
                    # Medians is in USD, so scale to INR
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
            
        # Evaluation Logic
        if submit_button:
            # 1. Convert inputs from INR to USD internally for the model
            loan_amnt_usd = float(loan_amnt_inr / CONVERSION_RATE)
            annual_inc_usd = float(annual_inc_inr / CONVERSION_RATE)
            installment_usd = float(installment_inr / CONVERSION_RATE)
            revol_bal_usd = float(revol_bal_inr / CONVERSION_RATE)
            
            # Start from feature medians to fill unspecified columns
            input_row = medians.copy()
            
            # 2. Insert converted values
            input_row['loan_amnt'] = loan_amnt_usd
            input_row['funded_amnt'] = loan_amnt_usd
            input_row['funded_amnt_inv'] = loan_amnt_usd
            input_row['int_rate'] = float(int_rate)
            input_row['installment'] = installment_usd
            input_row['annual_inc'] = annual_inc_usd
            input_row['dti'] = float(dti)
            
            # Domain-specific ratio (Currency units cancel out!)
            input_row['income_to_loan_ratio'] = float(annual_inc_usd / (loan_amnt_usd + 1e-5))
            
            # Numerical advanced
            input_row['open_acc'] = float(open_acc)
            input_row['total_acc'] = float(total_acc)
            input_row['revol_bal'] = revol_bal_usd
            input_row['revol_util'] = float(revol_util)
            input_row['delinq_2yrs'] = float(delinq_2yrs)
            input_row['inq_last_6mths'] = float(inq_last_6mths)
            input_row['pub_rec'] = float(pub_rec)
            input_row['pub_rec_bankruptcies'] = float(pub_rec_bankruptcies)
            input_row['tax_liens'] = float(tax_liens)
            
            # 3. Apply Winsorization limits (matching preprocessing.py)
            for col in winsorize_limits:
                if col in input_row:
                    limits = winsorize_limits[col]
                    input_row[col] = np.clip(input_row[col], limits['lower'], limits['upper'])
            
            # 4. Map Categorical strings to normalized training frequencies
            categorical_inputs = {
                'term': term,
                'grade': grade,
                'sub_grade': sub_grade,
                'emp_length': emp_length,
                'home_ownership': home_ownership,
                'verification_status': verification_status,
                'purpose': purpose,
                'application_type': application_type,
                'disbursement_method': disbursement_method,
                'initial_list_status': initial_list_status,
                'title': 'Debt consolidation' # Fallback default mode
            }
            
            for col, val in categorical_inputs.items():
                if col in cat_mappings:
                    mapping_dict = cat_mappings[col]['freq_map']
                    default_mode = cat_mappings[col]['default_mode']
                    
                    # Convert to string key
                    val_str = str(val)
                    if val_str in mapping_dict:
                        input_row[col] = mapping_dict[val_str]
                    else:
                        input_row[col] = mapping_dict.get(default_mode, 0.0)
            
            # 5. Convert to DataFrame matching training columns
            input_df = pd.DataFrame([input_row])
            input_df = input_df[features] # enforce column ordering
            
            # 6. Prediction
            pd_prob = float(model.predict_proba(input_df)[0, 1])
            
            # 7. Calculate Expected Loss (in INR)
            lgd = 0.50 # 50% Loss Given Default
            ead_inr = loan_amnt_inr # Exposure at Default in INR
            expected_loss_inr = pd_prob * lgd * ead_inr
            
            # Display Decision
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown("### Evaluation Results")
            
            dec_col1, dec_col2 = st.columns([1, 1])
            
            with dec_col1:
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
                    
                # Meter KPI Cards in INR
                sub_kpi1, sub_kpi2, sub_kpi3 = st.columns(3)
                with sub_kpi1:
                    st.metric(label="Probability of Default (PD)", value=f"{pd_prob * 100:.2f}%")
                with sub_kpi2:
                    el_display = f"₹{int(round(expected_loss_inr)):,}" if expected_loss_inr >= 100 else f"₹{expected_loss_inr:,.2f}"
                    st.metric(label="Expected Loss (EL)", value=el_display, help="EL = PD * LGD * EAD (assuming LGD=50%)")
                with sub_kpi3:
                    risk_cat = "Low" if pd_prob < 0.10 else "Medium" if pd_prob < 0.20 else "High" if pd_prob < 0.35 else "Critical"
                    st.metric(label="Risk Class", value=risk_cat)
            
            with dec_col2:
                # Gauge chart for probability
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
                
            # Explainability Section
            st.markdown("#### 🔍 Explain Decision (SHAP Contributions)")
            st.write("This interactive chart shows how each credit parameter pulled the applicant's risk score relative to the baseline bank risk.")
            
            # Compute SHAP
            shap_values = explainer(input_df)
            
            # Format SHAP values for Plotly
            shap_df = pd.DataFrame({
                'Feature': features,
                'SHAP Value': shap_values.values[0]
            })
            
            # Sort by absolute SHAP value
            shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
            shap_df = shap_df.sort_values(by='Abs SHAP', ascending=False).head(10)
            
            # Color coding (Positive = Increases risk = Red, Negative = Decreases risk = Green)
            shap_df['Color'] = np.where(shap_df['SHAP Value'] >= 0, '#ef4444', '#10b981')
            shap_df['Direction'] = np.where(shap_df['SHAP Value'] >= 0, 'Increases Risk 📈', 'Reduces Risk 📉')
            
            # Capitalize and clean feature names for display
            shap_df['Clean Feature'] = shap_df['Feature'].str.replace('_', ' ').str.title()
            
            # Plot
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
        st.write("Key metrics and statistics converted to Rupees based on historical credit records of 2.26 million borrowers.")
        
        hist_df = load_historical_sample()
        if hist_df is None:
            st.warning("Could not load historical dataset for dashboard charts.")
        else:
            # Reverse the grade map to string grade ('A', 'B', etc.)
            grade_freqs = cat_mappings['grade']['freq_map']
            rev_grade_map = {v: k for k, v in grade_freqs.items()}
            
            # Map grade and is_default
            df_plot = hist_df[['grade', 'is_default', 'int_rate', 'dti', 'annual_inc']].copy()
            df_plot['Grade Name'] = df_plot['grade'].map(rev_grade_map).fillna("Unknown")
            
            # Convert values to INR for visual representation
            df_plot['annual_inc_inr'] = df_plot['annual_inc'] * CONVERSION_RATE
            
            col_graph1, col_graph2 = st.columns(2)
            
            with col_graph1:
                st.markdown("#### The 12x Default Rate Gap Across Grades")
                st.write("Lower-grade loans (Grade G/F) exhibit exponentially higher default rates than Grade A loans.")
                
                # Compute default rate by grade
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
                    df_plot.sample(2000, random_state=42), # sample to keep plotly responsive
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
                
            # Distribution of Annual Income in INR
            st.markdown("#### Borrower Annual Income Distribution (Clipped at ₹2.25 Crores)")
            st.write("Histogram showing the distribution of annual income in Indian Rupees.")
            
            # Clip at 99th percentile for INR display
            max_inc_inr_display = 270000.0 * CONVERSION_RATE # Approx ₹2.25 Crores
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
            
            # Show metrics table
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
            
            # Plot model feature importance
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
